#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <mutex>
#include <numeric>
#include <optional>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace fs = std::filesystem;

namespace {

constexpr int kMachineCount = 5;
constexpr int kFamilyCount = 5;
constexpr std::array<int, kMachineCount> kMachineClass{0, 0, 1, 1, 2};
constexpr std::array<int, kMachineCount> kMachineCapacity{60, 60, 80, 80, 100};
constexpr std::array<double, kFamilyCount> kProcessingTime{190, 158, 202, 128, 104};
constexpr std::array<double, 3> kIpmTime{20, 30, 40};
constexpr std::array<double, 3> kPmTime{60, 90, 120};
constexpr std::array<double, 3> kMinimalRepairTime{80, 120, 160};
constexpr std::array<std::array<double, 2>, 3> kCorrectiveRange{{
    {20, 40}, {40, 60}, {60, 80}
}};
constexpr std::array<std::array<double, 2>, 3> kDegradationRange{{
    {0.001, 0.005}, {0.001, 0.003}, {0.001, 0.002}
}};
constexpr std::array<double, 3> kTerminalDegradationRate{0.005, 0.003, 0.002};
constexpr std::array<std::array<double, 2>, 3> kFailureRange{{
    {0.0001, 0.0003}, {0.0001, 0.0003}, {0.0001, 0.0003}
}};
constexpr std::array<double, 3> kFailureProportion{0.6, 0.3, 0.1};
constexpr std::array<std::array<double, 3>, 3> kIpmRecoveryRate{{
    {0.06931471805599453, 0.06019864021629608, 0.02153914580462271},
    {0.046209812037329684, 0.04013242681086405, 0.014359430536415141},
    {0.03465735902799726, 0.03009932010814804, 0.010769572902311356}
}};

enum class Maintenance : int { None = 0, Ipm = 1, Pm = 2 };
enum class InitialOrder : int { DueArrival = 0, ArrivalDue = 1, Slack = 2 };

struct Job {
    int id{};
    double arrival{};
    int family{};
    int size{};
    double due{};
};

struct Instance {
    int condition_index{};
    std::string condition_id;
    int instance_index{};
    std::uint64_t instance_seed{};
    double degradation_multiplier{1.0};
    fs::path job_file;
    std::string job_file_sha256;
    std::vector<Job> jobs;
};

struct BatchGene {
    std::vector<int> jobs;
    int family{};
    int machine{};
    Maintenance maintenance{Maintenance::None};
};

struct Operation {
    int sequence_index{};
    int machine{};
    std::string action;
    double start{};
    double end{};
    int family{-1};
    std::vector<int> jobs;
    int total_size{};
    double health_before{};
    double health_after{};
};

struct Evaluation {
    double total_tardiness{std::numeric_limits<double>::infinity()};
    double tie_break{std::numeric_limits<double>::infinity()};
    double makespan{};
    double total_completion{};
    double planned_maintenance_time{};
    std::vector<Operation> operations;
};

struct Individual {
    std::vector<BatchGene> genes;
    Evaluation evaluation;
};

struct Parameters {
    std::string search_strategy{"joint-neighborhood"};
    int refinement_generations{200};
    int population{100};
    int generations{2000};
    int local_search_individuals{1};
    int local_search_evaluations{500};
    int nonimproving_generations{200};
    int local_search_interval{20};
    int mutation_adaptation_generations{100};
    double mutation_probability{0.20};
    double formation_probability_ga{0.75};
    double formation_probability_ls{0.25};
    double local_search_deviation_percent{5.0};
    double arrival_window_factor{1.0};
    std::uint64_t planning_seed_base{40'000'000ULL};
    int threads{1};
    int instance_limit{0};
    bool resume{true};
    double degradation_multiplier{1.0};
};

struct GenerationRecord {
    int generation{};
    std::uint64_t evaluations{};
    double incumbent_ttd{};
    double population_best{};
    double population_mean{};
    double population_median{};
    double population_worst{};
    double population_std{};
    std::size_t unique_chromosomes{};
    int stagnant_generations{};
    int last_improvement_generation{};
    std::uint64_t local_search_calls{};
    std::uint64_t local_search_evaluations{};
    std::uint64_t local_search_improvements{};
    double effective_mutation_probability{};
    bool refinement_phase{};
};

struct SearchDiagnostics {
    double baseline_best_ttd{};
    int baseline_generations{};
    std::uint64_t baseline_evaluations{};
    double initial_best_ttd{};
    int last_improvement_generation{};
    std::string stop_reason;
    std::uint64_t local_search_calls{};
    std::uint64_t local_search_evaluations{};
    std::uint64_t local_search_improvements{};
    std::vector<GenerationRecord> convergence;
};

struct ResultSummary {
    int condition_index{};
    std::string condition_id;
    int instance_index{};
    std::uint64_t instance_seed{};
    std::uint64_t optimizer_seed{};
    std::string plan_file;
    double objective_ttd{};
    double surrogate_makespan{};
    double total_completion{};
    double planned_maintenance_time{};
    int batch_count{};
    int ipm_count{};
    int pm_count{};
    std::uint64_t evaluations{};
    int generations{};
    double initial_objective_ttd{};
    double improvement_percent{};
    int last_improvement_generation{};
    std::string stop_reason;
    std::string convergence_file;
    std::size_t final_unique_chromosomes{};
    double runtime_seconds{};
    std::string status;
    double baseline_best_ttd{};
    int baseline_generations{};
    std::uint64_t baseline_evaluations{};
};

std::vector<std::string> parse_csv_line(const std::string& line) {
    std::vector<std::string> fields;
    std::string current;
    bool quoted = false;
    for (std::size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (quoted) {
            if (c == '"' && i + 1 < line.size() && line[i + 1] == '"') {
                current.push_back('"');
                ++i;
            } else if (c == '"') {
                quoted = false;
            } else {
                current.push_back(c);
            }
        } else if (c == '"') {
            quoted = true;
        } else if (c == ',') {
            fields.push_back(current);
            current.clear();
        } else {
            current.push_back(c);
        }
    }
    if (quoted) throw std::runtime_error("Unterminated quoted CSV field.");
    fields.push_back(current);
    return fields;
}

std::string csv_escape(const std::string& value) {
    if (value.find_first_of(",\"\r\n") == std::string::npos) return value;
    std::string result = "\"";
    for (char c : value) {
        if (c == '"') result += "\"\"";
        else result.push_back(c);
    }
    result.push_back('"');
    return result;
}

std::unordered_map<std::string, std::size_t> header_map(
        const std::vector<std::string>& header) {
    std::unordered_map<std::string, std::size_t> result;
    for (std::size_t i = 0; i < header.size(); ++i) result.emplace(header[i], i);
    return result;
}

const std::string& required_field(
        const std::vector<std::string>& row,
        const std::unordered_map<std::string, std::size_t>& header,
        const std::string& name) {
    auto found = header.find(name);
    if (found == header.end() || found->second >= row.size()) {
        throw std::runtime_error("Missing CSV field: " + name);
    }
    return row[found->second];
}

std::vector<Instance> read_instance_manifest(
        const fs::path& dataset_dir, const fs::path& manifest_path) {
    std::ifstream input(manifest_path);
    if (!input) throw std::runtime_error("Cannot open instance manifest: " + manifest_path.string());
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("Empty instance manifest.");
    const auto header = header_map(parse_csv_line(line));
    std::vector<Instance> result;
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto row = parse_csv_line(line);
        Instance item;
        item.condition_index = std::stoi(required_field(row, header, "condition_index"));
        item.condition_id = required_field(row, header, "condition_id");
        item.instance_index = std::stoi(required_field(row, header, "instance_index"));
        item.instance_seed = std::stoull(required_field(row, header, "instance_seed"));
        item.degradation_multiplier = std::stod(required_field(row, header, "degradation_multiplier"));
        item.job_file = dataset_dir / fs::path(required_field(row, header, "job_file"));
        item.job_file_sha256 = required_field(row, header, "job_file_sha256");
        result.push_back(std::move(item));
    }
    std::sort(result.begin(), result.end(), [](const Instance& a, const Instance& b) {
        return std::tie(a.condition_index, a.instance_index)
             < std::tie(b.condition_index, b.instance_index);
    });
    return result;
}

void load_jobs(Instance& instance) {
    std::ifstream input(instance.job_file);
    if (!input) throw std::runtime_error("Cannot open job file: " + instance.job_file.string());
    std::string line;
    if (!std::getline(input, line)) throw std::runtime_error("Empty job file.");
    const auto header = header_map(parse_csv_line(line));
    while (std::getline(input, line)) {
        if (line.empty()) continue;
        const auto row = parse_csv_line(line);
        Job job;
        job.id = std::stoi(required_field(row, header, "job_id"));
        job.arrival = std::stod(required_field(row, header, "arrival_time"));
        job.family = std::stoi(required_field(row, header, "family"));
        job.size = std::stoi(required_field(row, header, "size"));
        job.due = std::stod(required_field(row, header, "due_time"));
        instance.jobs.push_back(job);
    }
    for (std::size_t i = 0; i < instance.jobs.size(); ++i) {
        const auto& job = instance.jobs[i];
        if (job.id != static_cast<int>(i) || job.family < 0 || job.family >= kFamilyCount
                || job.size <= 0 || job.size > 30 || !std::isfinite(job.arrival)
                || !std::isfinite(job.due)) {
            throw std::runtime_error("Frozen job record violates the C++ planner contract.");
        }
    }
}

int batch_size(const BatchGene& batch, const std::vector<Job>& jobs) {
    int result = 0;
    for (int id : batch.jobs) result += jobs.at(static_cast<std::size_t>(id)).size;
    return result;
}

double batch_release(const BatchGene& batch, const std::vector<Job>& jobs) {
    double result = 0.0;
    for (int id : batch.jobs) result = std::max(result, jobs.at(static_cast<std::size_t>(id)).arrival);
    return result;
}

double batch_due(const BatchGene& batch, const std::vector<Job>& jobs) {
    double result = std::numeric_limits<double>::infinity();
    for (int id : batch.jobs) result = std::min(result, jobs.at(static_cast<std::size_t>(id)).due);
    return result;
}

std::vector<int> feasible_machines(int size) {
    std::vector<int> result;
    for (int machine = 0; machine < kMachineCount; ++machine) {
        if (size <= kMachineCapacity[machine]) result.push_back(machine);
    }
    return result;
}

double lerp_state(const std::array<double, 4>& values, double health) {
    health = std::clamp(health, 0.0, 3.0);
    const int lower = std::min(2, static_cast<int>(std::floor(health)));
    const int upper = std::min(3, lower + 1);
    const double fraction = health - lower;
    return values[lower] * (1.0 - fraction) + values[upper] * fraction;
}

double expected_ipm_state_integer(int machine_class, int initial_state) {
    std::array<double, 4> probability{};
    probability.at(static_cast<std::size_t>(initial_state)) = 1.0;
    constexpr double dt = 0.02;
    double elapsed = 0.0;
    while (elapsed + 1e-12 < kIpmTime[machine_class]) {
        const double step = std::min(dt, kIpmTime[machine_class] - elapsed);
        auto next = probability;
        for (int state = 1; state <= 3; ++state) {
            const double transfer = probability[state]
                * (1.0 - std::exp(-kIpmRecoveryRate[machine_class][state - 1] * step));
            next[state] -= transfer;
            next[state - 1] += transfer;
        }
        probability = next;
        elapsed += step;
    }
    double expected = 0.0;
    for (int state = 0; state <= 3; ++state) expected += state * probability[state];
    return expected;
}

double expected_ipm_state(int machine_class, double health) {
    health = std::clamp(health, 0.0, 3.0);
    const int lower = static_cast<int>(std::floor(health));
    const int upper = static_cast<int>(std::ceil(health));
    if (lower == upper) return expected_ipm_state_integer(machine_class, lower);
    const double fraction = health - lower;
    return expected_ipm_state_integer(machine_class, lower) * (1.0 - fraction)
         + expected_ipm_state_integer(machine_class, upper) * fraction;
}

struct ProcessResult {
    double duration{};
    double final_health{};
};

ProcessResult expected_process(
        int machine_class, int family, double initial_health, double multiplier) {
    double health = std::clamp(initial_health, 0.0, 3.0);
    double remaining = kProcessingTime[family];
    double exposure = 0.0;
    double expected_downtime = 0.0;
    const auto [degradation_low, degradation_high] = kDegradationRange[machine_class];
    const auto [failure_low, failure_high] = kFailureRange[machine_class];
    const std::array<double, 4> degradation{
        degradation_low,
        (degradation_low + degradation_high) / 2.0,
        degradation_high,
        kTerminalDegradationRate[machine_class]
    };
    const std::array<double, 4> failure{
        failure_low,
        failure_low + (failure_high - failure_low) / 3.0,
        failure_low + 2.0 * (failure_high - failure_low) / 3.0,
        failure_high
    };
    const std::array<double, 4> production{1.0, 1.0 - 0.1 / 3.0, 1.0 - 0.2 / 3.0, 0.9};
    const double repair0 = kCorrectiveRange[machine_class][0];
    const double repair2 = kCorrectiveRange[machine_class][1];
    const double repair1 = (repair0 + repair2) / 2.0;
    const double expected_corrective = kFailureProportion[0] * repair0
        + kFailureProportion[1] * repair1 + kFailureProportion[2] * repair2;
    while (remaining > 1e-10) {
        const double rate = lerp_state(production, health);
        const double step = std::min(1.0, remaining / rate);
        remaining -= rate * step;
        const double degradation_rate = multiplier * lerp_state(degradation, health);
        const double failure_rate = multiplier * lerp_state(failure, health);
        expected_downtime += failure_rate * expected_corrective * step;
        if (health >= 3.0 - 1e-12) {
            expected_downtime += degradation_rate * kMinimalRepairTime[machine_class] * step;
        } else {
            health = std::min(3.0, health + degradation_rate * step);
        }
        exposure += step;
    }
    return {exposure + expected_downtime, health};
}

bool better(const Evaluation& a, const Evaluation& b) {
    if (a.total_tardiness < b.total_tardiness - 1e-9) return true;
    if (a.total_tardiness > b.total_tardiness + 1e-9) return false;
    return a.tie_break < b.tie_break - 1e-12;
}

bool primary_objective_improved(const Evaluation& a, const Evaluation& b) {
    return a.total_tardiness < b.total_tardiness - 1e-9;
}

class Optimizer {
public:
    Optimizer(const Instance& instance, const Parameters& parameters, std::uint64_t seed)
        : instance_(instance), p_(parameters), rng_(seed), optimizer_seed_(seed) {
        for (const auto& job : instance_.jobs) jobs_by_family_[job.family].push_back(job.id);
    }

    Individual run(int& generations_used) {
        if (p_.search_strategy == "joint-neighborhood") return run_refined(generations_used);
        auto initial = initial_solutions();
        std::sort(initial.begin(), initial.end(), individual_less);
        std::vector<Individual> population;
        population.reserve(static_cast<std::size_t>(p_.population));
        for (const auto& candidate : initial) {
            if (static_cast<int>(population.size()) >= p_.population) break;
            population.push_back(candidate);
        }
        while (static_cast<int>(population.size()) < p_.population) {
            Individual item = initial[population.size() % initial.size()];
            const int changes = 1 + static_cast<int>(population.size() % 5);
            for (int i = 0; i < changes; ++i) {
                mutate(item, (i + population.size()) % 2 == 0);
                repair(item);
            }
            item.evaluation = evaluate(item, false);
            ++evaluations_;
            population.push_back(std::move(item));
        }
        for (auto& item : population) {
            if (!std::isfinite(item.evaluation.total_tardiness)) {
                item.evaluation = evaluate(item, false);
                ++evaluations_;
            }
        }
        auto best_it = std::min_element(population.begin(), population.end(), individual_less);
        Individual global_best = *best_it;
        diagnostics_.initial_best_ttd = global_best.evaluation.total_tardiness;
        diagnostics_.last_improvement_generation = 0;
        int stagnant = 0;
        generations_used = 0;
        record_generation(population, global_best, 0, stagnant, p_.mutation_probability);
        for (int generation = 1; generation <= p_.generations; ++generation) {
            const double progress = std::min(
                1.0, static_cast<double>(stagnant)
                    / static_cast<double>(p_.mutation_adaptation_generations));
            const double effective_mutation = std::min(
                0.60, p_.mutation_probability * (1.0 + 2.0 * progress));
            const int matings = p_.population / 2;
            for (int mating = 0; mating < matings; ++mating) {
                int first = tournament(population);
                int second = tournament(population);
                for (int attempt = 0; second == first && attempt < 8; ++attempt) second = tournament(population);
                if (second == first) second = (first + 1) % p_.population;
                auto [child_a, child_b] = crossover(population[first], population[second]);
                // Formation crossover preserves job counts but arbitrary job
                // sizes can transiently exceed a receiver batch's capacity.
                // Restore feasibility before any assignment mutation samples
                // from that batch's machine set.
                repair(child_a);
                repair(child_b);
                if (uniform() < effective_mutation) mutate(child_a, uniform() < p_.formation_probability_ga);
                if (uniform() < effective_mutation) mutate(child_b, uniform() < p_.formation_probability_ga);
                repair(child_a);
                repair(child_b);
                child_a.evaluation = evaluate(child_a, false);
                child_b.evaluation = evaluate(child_b, false);
                evaluations_ += 2;
                std::array<Individual, 4> candidates{
                    population[first], population[second], std::move(child_a), std::move(child_b)
                };
                std::sort(candidates.begin(), candidates.end(), individual_less);
                population[first] = std::move(candidates[0]);
                population[second] = std::move(candidates[1]);
            }
            std::sort(population.begin(), population.end(), individual_less);
            if (p_.local_search_interval > 0
                    && generation % p_.local_search_interval == 0
                    && p_.local_search_individuals > 0
                    && p_.local_search_evaluations > 0) {
                const int count = std::min(p_.local_search_individuals, p_.population);
                for (int i = 0; i < count; ++i) {
                    const Evaluation before = population[i].evaluation;
                    const std::uint64_t evaluations_before = evaluations_;
                    local_search(population[i]);
                    ++diagnostics_.local_search_calls;
                    diagnostics_.local_search_evaluations += evaluations_ - evaluations_before;
                    diagnostics_.local_search_improvements += better(
                        population[i].evaluation, before);
                }
                std::sort(population.begin(), population.end(), individual_less);
            }
            const bool primary_improved = primary_objective_improved(
                population.front().evaluation, global_best.evaluation);
            if (better(population.front().evaluation, global_best.evaluation)) {
                global_best = population.front();
            }
            if (primary_improved) {
                stagnant = 0;
                diagnostics_.last_improvement_generation = generation;
            } else {
                ++stagnant;
            }
            generations_used = generation;
            record_generation(
                population, global_best, generation, stagnant,
                effective_mutation);
            if (stagnant >= p_.nonimproving_generations) {
                diagnostics_.stop_reason = "nonimproving_generation_limit";
                break;
            }
        }
        if (diagnostics_.stop_reason.empty()) {
            diagnostics_.stop_reason = "maximum_generations";
        }
        global_best.evaluation = evaluate(global_best, true);
        ++evaluations_;
        diagnostics_.baseline_best_ttd = global_best.evaluation.total_tardiness;
        diagnostics_.baseline_generations = generations_used;
        diagnostics_.baseline_evaluations = evaluations_;
        return global_best;
    }

    std::uint64_t evaluations() const { return evaluations_; }
    std::uint64_t optimizer_seed() const { return optimizer_seed_; }
    const SearchDiagnostics& diagnostics() const { return diagnostics_; }

private:
    Individual run_refined(int& generations_used) {
        // Apply joint-neighborhood refinement after the population search.
        Parameters baseline_parameters = p_;
        baseline_parameters.search_strategy = "legacy";
        Optimizer baseline(instance_, baseline_parameters, optimizer_seed_);
        Individual incumbent = baseline.run(generations_used);
        evaluations_ = baseline.evaluations_;
        diagnostics_ = std::move(baseline.diagnostics_);
        process_cache_ = std::move(baseline.process_cache_);
        rng_ = baseline.rng_;
        diagnostics_.baseline_best_ttd = incumbent.evaluation.total_tardiness;
        diagnostics_.baseline_generations = generations_used;
        diagnostics_.baseline_evaluations = evaluations_;
        if (p_.refinement_generations == 0) return incumbent;
        auto initial = initial_solutions();
        std::vector<Individual> population{incumbent};
        while (static_cast<int>(population.size()) < p_.population) {
            Individual candidate = population.size() % 3 == 0
                ? initial[random_int(0, static_cast<int>(initial.size()) - 1)] : incumbent;
            for (int step = 0, count = random_int(1, 5); step < count; ++step) {
                mutate(candidate, uniform() < 0.5);
                repair(candidate);
            }
            candidate.evaluation = evaluate(candidate, false);
            ++evaluations_;
            population.push_back(std::move(candidate));
        }
        diagnostics_.stop_reason = "refinement_generation_limit";
        int stagnant = 0;
        for (int step = 1; step <= p_.refinement_generations; ++step) {
            for (int mating = 0; mating < p_.population / 2; ++mating) {
                int a = tournament(population), b = tournament(population);
                if (a == b) b = (a + 1) % p_.population;
                auto [first, second] = crossover(population[a], population[b]);
                for (auto* child : {&first, &second}) {
                    repair(*child);
                    if (uniform() < 0.4) {
                        mutate(*child, uniform() < 0.5);
                        repair(*child);
                    }
                    child->evaluation = evaluate(*child, false);
                    ++evaluations_;
                }
                std::array<Individual, 4> family{
                    population[a], population[b], std::move(first), std::move(second)};
                std::sort(family.begin(), family.end(), individual_less);
                population[a] = std::move(family[0]);
                population[b] = std::move(family[1]);
            }
            std::sort(population.begin(), population.end(), individual_less);
            diversify(population, initial, stagnant, step);
            if (step == 1 || step % p_.local_search_interval == 0) {
                const int count = std::min(p_.local_search_individuals, p_.population);
                for (int i = 0; i < count; ++i) {
                    const auto before = population[i].evaluation;
                    const auto calls_before = evaluations_;
                    joint_local_search(population[i]);
                    ++diagnostics_.local_search_calls;
                    diagnostics_.local_search_evaluations += evaluations_ - calls_before;
                    diagnostics_.local_search_improvements += better(population[i].evaluation, before);
                }
                std::sort(population.begin(), population.end(), individual_less);
            }
            const bool improved = primary_objective_improved(population.front().evaluation, incumbent.evaluation);
            if (better(population.front().evaluation, incumbent.evaluation)) incumbent = population.front();
            stagnant = improved ? 0 : stagnant + 1;
            ++generations_used;
            if (improved) diagnostics_.last_improvement_generation = generations_used;
            record_generation(population, incumbent, generations_used, stagnant, 0.4);
            diagnostics_.convergence.back().refinement_phase = true;
        }
        incumbent.evaluation = evaluate(incumbent, true);
        ++evaluations_;
        return incumbent;
    }

    // Cache exact evaluations of the unchanged physical surrogate. No health
    // rounding, random draws, or simulator information is used by this cache.
    ProcessResult cached_process(int machine_class, int family, double health) const {
        const auto key = std::tuple(machine_class, family, health);
        auto found = process_cache_.find(key);
        if (found != process_cache_.end()) return found->second;
        const auto value = expected_process(
            machine_class, family, health, instance_.degradation_multiplier);
        if (process_cache_.size() >= 65536) process_cache_.clear();
        process_cache_.emplace(key, value);
        return value;
    }

    void diversify(std::vector<Individual>& population,
                   const std::vector<Individual>& initial, int stagnant, int generation) {
        // Canonical signatures ignore within-batch job permutations and
        // interleaving between independent machine queues.
        std::unordered_set<std::uint64_t> seen;
        for (int i = 0; i < p_.population; ++i) {
            const bool duplicate = !seen.insert(phenotype_signature(population[i])).second;
            const bool restart = stagnant > 0 && stagnant % 50 == 0
                && i >= 4 * p_.population / 5;
            if (i == 0 || (!duplicate && !restart)) continue;
            if (!restart && (generation % 5 != 0 || i < p_.population / 4)) continue;
            Individual candidate = restart
                ? initial[random_int(0, static_cast<int>(initial.size()) - 1)]
                : population[random_int(0, std::max(0, p_.population / 4 - 1))];
            for (int step = 0, count = random_int(2, 6); step < count; ++step) {
                mutate(candidate, uniform() < 0.5);
                repair(candidate);
            }
            candidate.evaluation = evaluate(candidate, false);
            ++evaluations_;
            seen.insert(phenotype_signature(candidate));
            population[i] = std::move(candidate);
        }
        std::sort(population.begin(), population.end(), individual_less);
    }

    std::uint64_t phenotype_signature(const Individual& item) const {
        Individual canonical;
        for (int machine = 0; machine < kMachineCount; ++machine) {
            for (auto batch : item.genes) {
                if (batch.machine != machine) continue;
                std::sort(batch.jobs.begin(), batch.jobs.end());
                canonical.genes.push_back(std::move(batch));
            }
        }
        return chromosome_signature(canonical);
    }

    std::uint64_t chromosome_signature(const Individual& item) const {
        std::uint64_t hash = 1469598103934665603ULL;
        auto mix = [&](std::uint64_t value) {
            hash ^= value;
            hash *= 1099511628211ULL;
        };
        for (const auto& batch : item.genes) {
            mix(0x9e3779b97f4a7c15ULL);
            mix(static_cast<std::uint64_t>(batch.family + 1));
            mix(static_cast<std::uint64_t>(batch.machine + 1));
            mix(static_cast<std::uint64_t>(static_cast<int>(batch.maintenance) + 1));
            for (int id : batch.jobs) mix(static_cast<std::uint64_t>(id + 1));
        }
        return hash;
    }

    void record_generation(
            const std::vector<Individual>& population,
            const Individual& global_best,
            int generation,
            int stagnant,
            double effective_mutation) {
        std::vector<double> objectives;
        objectives.reserve(population.size());
        std::unordered_set<std::uint64_t> unique;
        for (const auto& item : population) {
            objectives.push_back(item.evaluation.total_tardiness);
            unique.insert(chromosome_signature(item));
        }
        std::sort(objectives.begin(), objectives.end());
        const double mean = std::accumulate(
            objectives.begin(), objectives.end(), 0.0) / objectives.size();
        double variance = 0.0;
        for (double objective : objectives) {
            const double difference = objective - mean;
            variance += difference * difference;
        }
        variance /= objectives.size();
        const std::size_t middle = objectives.size() / 2;
        const double median = objectives.size() % 2
            ? objectives[middle]
            : (objectives[middle - 1] + objectives[middle]) / 2.0;
        diagnostics_.convergence.push_back(GenerationRecord{
            generation,
            evaluations_,
            global_best.evaluation.total_tardiness,
            objectives.front(),
            mean,
            median,
            objectives.back(),
            std::sqrt(variance),
            unique.size(),
            stagnant,
            diagnostics_.last_improvement_generation,
            diagnostics_.local_search_calls,
            diagnostics_.local_search_evaluations,
            diagnostics_.local_search_improvements,
            effective_mutation,
        });
    }

    static bool individual_less(const Individual& a, const Individual& b) {
        return better(a.evaluation, b.evaluation);
    }

    double uniform() { return std::generate_canonical<double, 53>(rng_); }

    int random_int(int lower, int upper) {
        if (lower > upper) {
            throw std::logic_error("Cannot sample an empty integer interval.");
        }
        std::uniform_int_distribution<int> distribution(lower, upper);
        return distribution(rng_);
    }

    template<class T>
    void shuffle(std::vector<T>& values) { std::shuffle(values.begin(), values.end(), rng_); }

    int choose_machine(int size, const std::array<double, kMachineCount>& available,
                       double release, int family, bool capacity_aware) {
        int best_machine = -1;
        double best_score = std::numeric_limits<double>::infinity();
        for (int machine : feasible_machines(size)) {
            const double completion = std::max(available[machine], release) + kProcessingTime[family];
            const double capacity_penalty = capacity_aware
                ? kProcessingTime[family]
                    * static_cast<double>(kMachineCapacity[machine] - size) / 100.0
                : 0.0;
            const double score = completion + capacity_penalty;
            if (score < best_score - 1e-12) {
                best_score = score;
                best_machine = machine;
            }
        }
        if (best_machine < 0) throw std::runtime_error("No machine can process a batch.");
        return best_machine;
    }

    std::vector<Individual> initial_solutions() {
        auto build = [&](int target_capacity, InitialOrder order,
                         bool arrival_aware, bool capacity_aware) {
            Individual result;
            for (int family = 0; family < kFamilyCount; ++family) {
                auto ids = jobs_by_family_[family];
                std::sort(ids.begin(), ids.end(), [&](int a, int b) {
                    const Job& ja = instance_.jobs[a];
                    const Job& jb = instance_.jobs[b];
                    if (order == InitialOrder::ArrivalDue) {
                        return std::tie(ja.arrival, ja.due, ja.id)
                             < std::tie(jb.arrival, jb.due, jb.id);
                    }
                    if (order == InitialOrder::Slack) {
                        const double slack_a = ja.due - ja.arrival - kProcessingTime[family];
                        const double slack_b = jb.due - jb.arrival - kProcessingTime[family];
                        return std::tie(slack_a, ja.due, ja.arrival, ja.id)
                             < std::tie(slack_b, jb.due, jb.arrival, jb.id);
                    }
                    return std::tie(ja.due, ja.arrival, ja.id)
                         < std::tie(jb.due, jb.arrival, jb.id);
                });
                BatchGene current;
                current.family = family;
                current.machine = 4;
                for (int id : ids) {
                    const int next_size = batch_size(current, instance_.jobs)
                                        + instance_.jobs[id].size;
                    double arrival_low = instance_.jobs[id].arrival;
                    double arrival_high = instance_.jobs[id].arrival;
                    for (int current_id : current.jobs) {
                        arrival_low = std::min(arrival_low, instance_.jobs[current_id].arrival);
                        arrival_high = std::max(arrival_high, instance_.jobs[current_id].arrival);
                    }
                    const bool exceeds_arrival_window = arrival_aware
                        && !current.jobs.empty()
                        && arrival_high - arrival_low
                            > p_.arrival_window_factor * kProcessingTime[family];
                    if (!current.jobs.empty()
                            && (next_size > target_capacity || exceeds_arrival_window)) {
                        result.genes.push_back(std::move(current));
                        current = BatchGene{};
                        current.family = family;
                        current.machine = 4;
                    }
                    current.jobs.push_back(id);
                }
                if (!current.jobs.empty()) result.genes.push_back(std::move(current));
            }
            std::sort(result.genes.begin(), result.genes.end(), [&](const BatchGene& a, const BatchGene& b) {
                return std::tuple(batch_due(a, instance_.jobs), batch_release(a, instance_.jobs), a.family)
                     < std::tuple(batch_due(b, instance_.jobs), batch_release(b, instance_.jobs), b.family);
            });
            std::array<double, kMachineCount> available{};
            for (auto& batch : result.genes) {
                const int size = batch_size(batch, instance_.jobs);
                batch.machine = choose_machine(
                    size, available, batch_release(batch, instance_.jobs),
                    batch.family, capacity_aware);
                available[batch.machine] = std::max(
                    available[batch.machine], batch_release(batch, instance_.jobs))
                    + kProcessingTime[batch.family];
            }
            repair(result);
            result.evaluation = evaluate(result, false);
            ++evaluations_;
            return result;
        };
        std::vector<Individual> candidates;
        for (int capacity : {60, 80, 100}) {
            for (InitialOrder order : {
                    InitialOrder::DueArrival,
                    InitialOrder::ArrivalDue,
                    InitialOrder::Slack}) {
                for (bool arrival_aware : {false, true}) {
                    for (bool capacity_aware : {false, true}) {
                        candidates.push_back(build(
                            capacity, order, arrival_aware, capacity_aware));
                    }
                }
            }
        }
        return candidates;
    }

    void formation_mutation(Individual& item) {
        std::vector<int> families;
        for (int family = 0; family < kFamilyCount; ++family) {
            int count = 0;
            for (const auto& batch : item.genes) count += batch.family == family;
            if (count >= 1) families.push_back(family);
        }
        if (families.empty()) return;
        const int family = families[random_int(0, static_cast<int>(families.size()) - 1)];
        std::vector<int> positions;
        for (int i = 0; i < static_cast<int>(item.genes.size()); ++i) {
            if (item.genes[i].family == family) positions.push_back(i);
        }
        const int operation = random_int(0, 3);
        if (operation == 0 && positions.size() >= 2) {
            int a = positions[random_int(0, static_cast<int>(positions.size()) - 1)];
            int b = positions[random_int(0, static_cast<int>(positions.size()) - 1)];
            if (a == b) b = positions[(std::find(positions.begin(), positions.end(), a) - positions.begin() + 1) % positions.size()];
            auto& first = item.genes[a];
            auto& second = item.genes[b];
            const int ia = random_int(0, static_cast<int>(first.jobs.size()) - 1);
            const int ib = random_int(0, static_cast<int>(second.jobs.size()) - 1);
            std::swap(first.jobs[ia], second.jobs[ib]);
        } else if (operation == 1 && positions.size() >= 2) {
            const int source_position = positions[random_int(0, static_cast<int>(positions.size()) - 1)];
            int target_position = positions[random_int(0, static_cast<int>(positions.size()) - 1)];
            if (source_position == target_position) return;
            auto& source = item.genes[source_position];
            auto& target = item.genes[target_position];
            if (source.jobs.size() <= 1) return;
            const int index = random_int(0, static_cast<int>(source.jobs.size()) - 1);
            target.jobs.push_back(source.jobs[index]);
            source.jobs.erase(source.jobs.begin() + index);
        } else if (operation == 2) {
            const int position = positions[random_int(0, static_cast<int>(positions.size()) - 1)];
            auto& source = item.genes[position];
            if (source.jobs.size() <= 1) return;
            shuffle(source.jobs);
            BatchGene split = source;
            split.jobs.assign(source.jobs.begin() + source.jobs.size() / 2, source.jobs.end());
            source.jobs.erase(source.jobs.begin() + source.jobs.size() / 2, source.jobs.end());
            item.genes.insert(item.genes.begin() + position + 1, std::move(split));
        } else if (positions.size() >= 2) {
            int a = positions[random_int(0, static_cast<int>(positions.size()) - 1)];
            int b = positions[random_int(0, static_cast<int>(positions.size()) - 1)];
            if (a == b) return;
            if (a > b) std::swap(a, b);
            auto merged = item.genes[a];
            merged.jobs.insert(merged.jobs.end(), item.genes[b].jobs.begin(), item.genes[b].jobs.end());
            if (batch_size(merged, instance_.jobs) <= 100) {
                item.genes[a] = std::move(merged);
                item.genes.erase(item.genes.begin() + b);
            }
        }
    }

    void load_balancing_mutation(Individual& item) {
        if (item.genes.empty()) return;
        std::array<double, kMachineCount> workload{};
        for (const auto& batch : item.genes) {
            workload[batch.machine] += kProcessingTime[batch.family];
            if (batch.maintenance == Maintenance::Ipm) {
                workload[batch.machine] += kIpmTime[kMachineClass[batch.machine]];
            } else if (batch.maintenance == Maintenance::Pm) {
                workload[batch.machine] += kPmTime[kMachineClass[batch.machine]];
            }
        }
        const int busiest = static_cast<int>(std::distance(
            workload.begin(), std::max_element(workload.begin(), workload.end())));
        std::vector<int> movable;
        for (int index = 0; index < static_cast<int>(item.genes.size()); ++index) {
            const auto& batch = item.genes[index];
            if (batch.machine != busiest) continue;
            const auto machines = feasible_machines(batch_size(batch, instance_.jobs));
            if (machines.size() > 1) movable.push_back(index);
        }
        if (movable.empty()) return;
        const int index = movable[random_int(0, static_cast<int>(movable.size()) - 1)];
        auto machines = feasible_machines(batch_size(item.genes[index], instance_.jobs));
        std::sort(machines.begin(), machines.end(), [&](int a, int b) {
            return std::tie(workload[a], kMachineCapacity[a], a)
                 < std::tie(workload[b], kMachineCapacity[b], b);
        });
        for (int machine : machines) {
            if (machine != busiest) {
                item.genes[index].machine = machine;
                return;
            }
        }
    }

    void same_machine_sequence_mutation(Individual& item) {
        std::vector<int> machines;
        for (int machine = 0; machine < kMachineCount; ++machine) {
            int count = 0;
            for (const auto& batch : item.genes) count += batch.machine == machine;
            if (count >= 2) machines.push_back(machine);
        }
        if (machines.empty()) return;
        const int machine = machines[random_int(0, static_cast<int>(machines.size()) - 1)];
        std::vector<int> positions;
        for (int index = 0; index < static_cast<int>(item.genes.size()); ++index) {
            if (item.genes[index].machine == machine) positions.push_back(index);
        }
        int first = random_int(0, static_cast<int>(positions.size()) - 1);
        int second = random_int(0, static_cast<int>(positions.size()) - 1);
        if (first == second) second = (second + 1) % static_cast<int>(positions.size());
        std::swap(item.genes[positions[first]], item.genes[positions[second]]);
    }

    void arrival_span_split_mutation(Individual& item) {
        std::vector<int> candidates;
        for (int index = 0; index < static_cast<int>(item.genes.size()); ++index) {
            const auto& batch = item.genes[index];
            if (batch.jobs.size() < 2) continue;
            double low = std::numeric_limits<double>::infinity();
            double high = 0.0;
            for (int id : batch.jobs) {
                low = std::min(low, instance_.jobs[id].arrival);
                high = std::max(high, instance_.jobs[id].arrival);
            }
            if (high - low > 0.25 * p_.arrival_window_factor
                    * kProcessingTime[batch.family]) {
                candidates.push_back(index);
            }
        }
        if (candidates.empty()) return;
        const int position = candidates[random_int(0, static_cast<int>(candidates.size()) - 1)];
        auto& source = item.genes[position];
        std::sort(source.jobs.begin(), source.jobs.end(), [&](int a, int b) {
            return std::tie(instance_.jobs[a].arrival, instance_.jobs[a].due, a)
                 < std::tie(instance_.jobs[b].arrival, instance_.jobs[b].due, b);
        });
        int split_index = 1;
        double largest_gap = -1.0;
        for (int index = 1; index < static_cast<int>(source.jobs.size()); ++index) {
            const double gap = instance_.jobs[source.jobs[index]].arrival
                             - instance_.jobs[source.jobs[index - 1]].arrival;
            if (gap > largest_gap) {
                largest_gap = gap;
                split_index = index;
            }
        }
        BatchGene split = source;
        split.jobs.assign(source.jobs.begin() + split_index, source.jobs.end());
        source.jobs.erase(source.jobs.begin() + split_index, source.jobs.end());
        item.genes.insert(item.genes.begin() + position + 1, std::move(split));
    }

    void sequence_or_gene_mutation(Individual& item) {
        if (item.genes.empty()) return;
        const int operation = random_int(0, 4);
        const int index = random_int(0, static_cast<int>(item.genes.size()) - 1);
        if (operation == 0) {
            same_machine_sequence_mutation(item);
        } else if (operation == 1) {
            auto machines = feasible_machines(batch_size(item.genes[index], instance_.jobs));
            item.genes[index].machine = machines[random_int(0, static_cast<int>(machines.size()) - 1)];
        } else if (operation == 2) {
            item.genes[index].maintenance = static_cast<Maintenance>(random_int(0, 2));
        } else if (operation == 3) {
            load_balancing_mutation(item);
        } else {
            arrival_span_split_mutation(item);
        }
    }

    void mutate(Individual& item, bool formation) {
        if (formation) formation_mutation(item);
        else sequence_or_gene_mutation(item);
        item.evaluation = Evaluation{};
    }

    std::vector<int> order_crossover(
            const std::vector<int>& receiver, const std::vector<int>& donor) {
        if (receiver.size() < 2 || receiver.size() != donor.size()) return receiver;
        const int left = random_int(0, static_cast<int>(receiver.size()) - 2);
        const int right = random_int(left + 1, static_cast<int>(receiver.size()) - 1);
        std::vector<int> result(receiver.size(), -1);
        std::unordered_set<int> used;
        for (int i = left; i <= right; ++i) {
            result[i] = receiver[i];
            used.insert(receiver[i]);
        }
        int cursor = (right + 1) % static_cast<int>(receiver.size());
        for (std::size_t offset = 0; offset < donor.size(); ++offset) {
            const int value = donor[(right + 1 + offset) % donor.size()];
            if (used.contains(value)) continue;
            while (result[cursor] != -1) cursor = (cursor + 1) % static_cast<int>(result.size());
            result[cursor] = value;
            used.insert(value);
        }
        return result;
    }

    void replace_family_order(
            Individual& child, const Individual& donor, int family) {
        std::vector<int> receiver_jobs;
        std::vector<int> donor_jobs;
        std::vector<int> positions;
        std::vector<int> target_counts;
        for (int i = 0; i < static_cast<int>(child.genes.size()); ++i) {
            if (child.genes[i].family == family) {
                positions.push_back(i);
                target_counts.push_back(static_cast<int>(child.genes[i].jobs.size()));
                receiver_jobs.insert(receiver_jobs.end(), child.genes[i].jobs.begin(), child.genes[i].jobs.end());
            }
        }
        for (const auto& batch : donor.genes) {
            if (batch.family == family) donor_jobs.insert(donor_jobs.end(), batch.jobs.begin(), batch.jobs.end());
        }
        if (receiver_jobs.size() < 2 || receiver_jobs.size() != donor_jobs.size()) return;
        auto crossed = order_crossover(receiver_jobs, donor_jobs);
        std::size_t cursor = 0;
        for (std::size_t i = 0; i < positions.size(); ++i) {
            auto& jobs = child.genes[positions[i]].jobs;
            jobs.assign(crossed.begin() + static_cast<std::ptrdiff_t>(cursor),
                        crossed.begin() + static_cast<std::ptrdiff_t>(cursor + target_counts[i]));
            cursor += static_cast<std::size_t>(target_counts[i]);
        }
    }

    void sequence_crossover(Individual& child, const Individual& donor) {
        if (child.genes.size() < 2 || donor.genes.empty()) return;
        std::array<int, kFamilyCount> donor_rank{};
        donor_rank.fill(std::numeric_limits<int>::max());
        for (int i = 0; i < static_cast<int>(donor.genes.size()); ++i) {
            donor_rank[donor.genes[i].family] = std::min(donor_rank[donor.genes[i].family], i);
        }
        const int left = random_int(0, static_cast<int>(child.genes.size()) - 2);
        const int right = random_int(left + 1, static_cast<int>(child.genes.size()) - 1);
        std::stable_sort(child.genes.begin() + left, child.genes.begin() + right + 1,
            [&](const BatchGene& a, const BatchGene& b) {
                return donor_rank[a.family] < donor_rank[b.family];
            });
    }

    std::pair<Individual, Individual> crossover(const Individual& first, const Individual& second) {
        if (p_.search_strategy == "joint-neighborhood" && uniform() < 0.5) {
            return {batch_preserving_child(first, second), batch_preserving_child(second, first)};
        }
        Individual a = first;
        Individual b = second;
        if (uniform() < p_.formation_probability_ga) {
            const int family = random_int(0, kFamilyCount - 1);
            replace_family_order(a, second, family);
            replace_family_order(b, first, family);
        } else {
            sequence_crossover(a, second);
            sequence_crossover(b, first);
        }
        const std::size_t overlap = std::min(a.genes.size(), b.genes.size());
        for (std::size_t i = 0; i < overlap; ++i) {
            if (uniform() < 0.5) {
                const int machine_a = a.genes[i].machine;
                const int machine_b = b.genes[i].machine;
                if (batch_size(a.genes[i], instance_.jobs) <= kMachineCapacity[machine_b]) a.genes[i].machine = machine_b;
                if (batch_size(b.genes[i], instance_.jobs) <= kMachineCapacity[machine_a]) b.genes[i].machine = machine_a;
            }
            if (uniform() < 0.5) std::swap(a.genes[i].maintenance, b.genes[i].maintenance);
        }
        a.evaluation = Evaluation{};
        b.evaluation = Evaluation{};
        return {std::move(a), std::move(b)};
    }

    Individual batch_preserving_child(const Individual& receiver, const Individual& donor) {
        if (receiver.genes.empty()) return donor;
        const int left = random_int(0, static_cast<int>(receiver.genes.size()) - 1);
        const int right = random_int(left, static_cast<int>(receiver.genes.size()) - 1);
        std::vector<bool> inherited(instance_.jobs.size(), false);
        std::vector<BatchGene> block;
        for (int i = left; i <= right; ++i) {
            block.push_back(receiver.genes[i]);
            for (int id : receiver.genes[i].jobs) inherited[id] = true;
        }
        Individual child;
        for (auto batch : donor.genes) {
            std::erase_if(batch.jobs, [&](int id) { return inherited[id]; });
            if (!batch.jobs.empty()) child.genes.push_back(std::move(batch));
        }
        const auto position = child.genes.begin()
            + std::min<std::size_t>(left, child.genes.size());
        child.genes.insert(position, block.begin(), block.end());
        return child;
    }

    void repair(Individual& item) {
        std::vector<bool> seen(instance_.jobs.size(), false);
        std::vector<BatchGene> repaired;
        for (auto batch : item.genes) {
            std::vector<int> valid;
            for (int id : batch.jobs) {
                if (id < 0 || id >= static_cast<int>(instance_.jobs.size()) || seen[id]) continue;
                if (instance_.jobs[id].family != batch.family) continue;
                seen[id] = true;
                valid.push_back(id);
            }
            if (valid.empty()) continue;
            batch.jobs = std::move(valid);
            while (!batch.jobs.empty()) {
                BatchGene piece = batch;
                piece.jobs.clear();
                int size = 0;
                auto cursor = batch.jobs.begin();
                while (cursor != batch.jobs.end()) {
                    const int next = instance_.jobs[*cursor].size;
                    if (size + next <= 100) {
                        size += next;
                        piece.jobs.push_back(*cursor);
                        cursor = batch.jobs.erase(cursor);
                    } else {
                        ++cursor;
                    }
                }
                if (piece.jobs.empty()) throw std::runtime_error("Job exceeds maximum machine capacity.");
                auto machines = feasible_machines(size);
                if (std::find(machines.begin(), machines.end(), piece.machine) == machines.end()) {
                    piece.machine = machines.front();
                }
                repaired.push_back(std::move(piece));
            }
        }
        for (const auto& job : instance_.jobs) {
            if (!seen[job.id]) {
                repaired.push_back(BatchGene{{job.id}, job.family, 0, Maintenance::None});
                auto machines = feasible_machines(job.size);
                repaired.back().machine = machines.front();
            }
        }
        item.genes = std::move(repaired);
    }

    Evaluation evaluate(const Individual& item, bool keep_operations) const {
        Evaluation result;
        result.total_tardiness = 0.0;
        result.total_completion = 0.0;
        std::vector<double> completion(instance_.jobs.size(), 0.0);
        std::array<std::vector<int>, kMachineCount> queues;
        for (int i = 0; i < static_cast<int>(item.genes.size()); ++i) {
            queues[item.genes[i].machine].push_back(i);
        }
        for (int machine = 0; machine < kMachineCount; ++machine) {
            double clock = 0.0;
            double health = 0.0;
            auto pending = queues[machine];
            while (!pending.empty()) {
                int chosen_position = 0;
                const BatchGene& head = item.genes[pending.front()];
                const double head_release = batch_release(head, instance_.jobs);
                if (head_release > clock + 1e-12) {
                    double best_duration = std::numeric_limits<double>::infinity();
                    for (int position = 1; position < static_cast<int>(pending.size()); ++position) {
                        const auto& candidate = item.genes[pending[position]];
                        const double release = batch_release(candidate, instance_.jobs);
                        if (release > clock + 1e-12) continue;
                        double trial_health = health;
                        double maintenance_duration = 0.0;
                        if (candidate.maintenance == Maintenance::Ipm) {
                            maintenance_duration = kIpmTime[kMachineClass[machine]];
                            trial_health = expected_ipm_state(kMachineClass[machine], trial_health);
                        } else if (candidate.maintenance == Maintenance::Pm) {
                            maintenance_duration = kPmTime[kMachineClass[machine]];
                            trial_health = 0.0;
                        }
                        const auto process = cached_process(
                            kMachineClass[machine], candidate.family, trial_health);
                        const double duration = maintenance_duration + process.duration;
                        if (clock + duration <= head_release + 1e-12 && duration < best_duration) {
                            chosen_position = position;
                            best_duration = duration;
                        }
                    }
                }
                const int gene_index = pending[chosen_position];
                pending.erase(pending.begin() + chosen_position);
                const auto& batch = item.genes[gene_index];
                const double release = batch_release(batch, instance_.jobs);
                if (batch.maintenance != Maintenance::None) {
                    const double before = health;
                    const double duration = batch.maintenance == Maintenance::Ipm
                        ? kIpmTime[kMachineClass[machine]] : kPmTime[kMachineClass[machine]];
                    const double start = clock;
                    clock += duration;
                    result.planned_maintenance_time += duration;
                    health = batch.maintenance == Maintenance::Ipm
                        ? expected_ipm_state(kMachineClass[machine], health) : 0.0;
                    if (keep_operations) {
                        result.operations.push_back(Operation{
                            gene_index, machine,
                            batch.maintenance == Maintenance::Ipm ? "IPM" : "PM",
                            start, clock, -1, {}, 0, before, health
                        });
                    }
                }
                const double start = std::max(clock, release);
                const double before = health;
                const auto process = cached_process(
                    kMachineClass[machine], batch.family, health);
                clock = start + process.duration;
                health = process.final_health;
                const int size = batch_size(batch, instance_.jobs);
                for (int id : batch.jobs) completion[id] = clock;
                if (keep_operations) {
                    result.operations.push_back(Operation{
                        gene_index, machine, "PROCESS", start, clock, batch.family,
                        batch.jobs, size, before, health
                    });
                }
            }
            result.makespan = std::max(result.makespan, clock);
        }
        for (const auto& job : instance_.jobs) {
            result.total_tardiness += std::max(0.0, completion[job.id] - job.due);
            result.total_completion += completion[job.id];
        }
        result.tie_break = result.total_completion + 1e-3 * result.planned_maintenance_time;
        if (keep_operations) {
            std::sort(result.operations.begin(), result.operations.end(), [](const Operation& a, const Operation& b) {
                return std::tie(a.start, a.machine, a.sequence_index, a.action)
                     < std::tie(b.start, b.machine, b.sequence_index, b.action);
            });
        }
        return result;
    }

    int tournament(const std::vector<Individual>& population) {
        const int a = random_int(0, static_cast<int>(population.size()) - 1);
        const int b = random_int(0, static_cast<int>(population.size()) - 1);
        return individual_less(population[a], population[b]) ? a : b;
    }

    void local_search(Individual& item) {
        Individual current = item;
        Individual best_local = item;
        for (int evaluation = 0; evaluation < p_.local_search_evaluations; ++evaluation) {
            Individual neighbor = current;
            mutate(neighbor, uniform() < p_.formation_probability_ls);
            repair(neighbor);
            neighbor.evaluation = evaluate(neighbor, false);
            ++evaluations_;
            if (better(neighbor.evaluation, best_local.evaluation)) {
                best_local = neighbor;
                current = std::move(neighbor);
            } else {
                const double threshold = best_local.evaluation.total_tardiness
                    * (1.0 + p_.local_search_deviation_percent / 100.0);
                if (neighbor.evaluation.total_tardiness <= threshold + 1e-9) {
                    current = std::move(neighbor);
                } else {
                    current = best_local;
                }
            }
        }
        item = std::move(best_local);
    }

    void joint_local_search(Individual& item) {
        // Six complementary neighborhoods receive the same trial allocation.
        // Every accepted step improves the unchanged lexicographic objective.
        int remaining = p_.local_search_evaluations;
        int empty_cycles = 0;
        while (remaining > 0 && empty_cycles < 2) {
            bool improved_cycle = false;
            const int offset = random_int(0, 5);
            for (int phase = 0; phase < 6 && remaining > 0; ++phase) {
                const int kind = (phase + offset) % 6;
                const int quota = std::min(remaining,
                    std::max(1, p_.local_search_evaluations / 12));
                int used = 0;
                Individual best = item;
                std::unordered_set<std::uint64_t> visited;
                visited.insert(phenotype_signature(item));
                auto test = [&](Individual candidate) {
                    if (used >= quota || remaining <= 0) return;
                    repair(candidate);
                    if (!visited.insert(phenotype_signature(candidate)).second) return;
                    candidate.evaluation = evaluate(candidate, false);
                    ++evaluations_; ++used; --remaining;
                    if (better(candidate.evaluation, best.evaluation)) best = std::move(candidate);
                };
                std::vector<int> order(item.genes.size());
                std::iota(order.begin(), order.end(), 0);
                shuffle(order);
                for (int index : order) {
                    if (used >= quota || remaining <= 0) break;
                    const auto& batch = item.genes[index];
                    if (kind == 0) {
                        // Explicit maintenance coordinate search, including deletion.
                        for (int action = 0; action < 3; ++action) {
                            if (action == static_cast<int>(batch.maintenance)) continue;
                            Individual neighbor = item;
                            neighbor.genes[index].maintenance = static_cast<Maintenance>(action);
                            test(std::move(neighbor));
                        }
                    } else if (kind == 1) {
                        // Machine and maintenance are optimized jointly.
                        for (int machine : feasible_machines(batch_size(batch, instance_.jobs))) {
                            for (int action = 0; action < 3; ++action) {
                                Individual neighbor = item;
                                neighbor.genes[index].machine = machine;
                                neighbor.genes[index].maintenance = static_cast<Maintenance>(action);
                                test(std::move(neighbor));
                            }
                        }
                    } else if (kind == 2) {
                        std::vector<int> positions(item.genes.size());
                        std::iota(positions.begin(), positions.end(), 0);
                        shuffle(positions);
                        for (int target : positions) {
                            if (target == index || used >= quota) continue;
                            Individual neighbor = item;
                            auto moving = neighbor.genes[index];
                            neighbor.genes.erase(neighbor.genes.begin() + index);
                            neighbor.genes.insert(neighbor.genes.begin() + target, std::move(moving));
                            test(std::move(neighbor));
                        }
                    } else if (kind == 3) {
                        // Exchange maintenance locations without changing the batches.
                        for (int target : order) {
                            if (target == index || item.genes[target].machine != batch.machine
                                    || used >= quota) continue;
                            Individual neighbor = item;
                            std::swap(neighbor.genes[index].maintenance,
                                      neighbor.genes[target].maintenance);
                            test(std::move(neighbor));
                        }
                    } else if (kind == 4) {
                        // Joint merge and assignment avoids repair-driven assignments.
                        for (int target : order) {
                            if (target == index || item.genes[target].family != batch.family
                                    || used >= quota) continue;
                            const int size = batch_size(batch, instance_.jobs)
                                + batch_size(item.genes[target], instance_.jobs);
                            auto machines = feasible_machines(size);
                            for (int machine : machines) {
                                for (int action = 0; action < 3; ++action) {
                                    Individual neighbor = item;
                                    auto merged = batch;
                                    merged.jobs.insert(merged.jobs.end(),
                                        item.genes[target].jobs.begin(), item.genes[target].jobs.end());
                                    merged.machine = machine;
                                    merged.maintenance = static_cast<Maintenance>(action);
                                    neighbor.genes[index] = std::move(merged);
                                    neighbor.genes.erase(neighbor.genes.begin() + target);
                                    test(std::move(neighbor));
                                }
                            }
                        }
                    } else {
                        // Single-job insertion and extraction change batch boundaries.
                        auto ids = batch.jobs;
                        shuffle(ids);
                        for (int id : ids) {
                            if (batch.jobs.size() <= 1 || used >= quota) break;
                            for (int target : order) {
                                if (target == index || item.genes[target].family != batch.family
                                        || used >= quota) continue;
                                if (batch_size(item.genes[target], instance_.jobs)
                                        + instance_.jobs[id].size
                                        > kMachineCapacity[item.genes[target].machine]) continue;
                                Individual neighbor = item;
                                std::erase(neighbor.genes[index].jobs, id);
                                neighbor.genes[target].jobs.push_back(id);
                                test(std::move(neighbor));
                            }
                            if (used >= quota) break;
                            Individual neighbor = item;
                            std::erase(neighbor.genes[index].jobs, id);
                            BatchGene extracted{{id}, batch.family, batch.machine, Maintenance::None};
                            neighbor.genes.insert(neighbor.genes.begin() + index, std::move(extracted));
                            test(std::move(neighbor));
                        }
                    }
                }
                if (better(best.evaluation, item.evaluation)) {
                    item = std::move(best);
                    improved_cycle = true;
                }
            }
            if (improved_cycle) empty_cycles = 0;
            else ++empty_cycles;
        }
    }

    void validate(const Individual& item) const {
        std::vector<int> count(instance_.jobs.size(), 0);
        for (const auto& batch : item.genes) {
            if (batch.jobs.empty() || batch.machine < 0 || batch.machine >= kMachineCount) {
                throw std::runtime_error("Invalid batch in optimized plan.");
            }
            if (batch_size(batch, instance_.jobs) > kMachineCapacity[batch.machine]) {
                throw std::runtime_error("Optimized batch exceeds assigned machine capacity.");
            }
            for (int id : batch.jobs) {
                if (instance_.jobs[id].family != batch.family) {
                    throw std::runtime_error("Optimized batch mixes incompatible families.");
                }
                ++count[id];
            }
        }
        if (std::any_of(count.begin(), count.end(), [](int value) { return value != 1; })) {
            throw std::runtime_error("Optimized plan does not cover every job exactly once.");
        }
    }

    const Instance& instance_;
    const Parameters& p_;
    std::mt19937_64 rng_;
    std::uint64_t optimizer_seed_{};
    std::uint64_t evaluations_{};
    SearchDiagnostics diagnostics_;
    mutable std::map<std::tuple<int, int, double>, ProcessResult> process_cache_;
    std::array<std::vector<int>, kFamilyCount> jobs_by_family_;
};

std::string join_jobs(const std::vector<int>& jobs) {
    std::ostringstream stream;
    for (std::size_t i = 0; i < jobs.size(); ++i) {
        if (i) stream << ';';
        stream << jobs[i];
    }
    return stream.str();
}

void atomic_replace(const fs::path& temporary, const fs::path& target) {
    std::error_code error;
    fs::remove(target, error);
    error.clear();
    fs::rename(temporary, target, error);
    if (error) throw std::runtime_error("Cannot atomically publish " + target.string() + ": " + error.message());
}

void write_convergence(
        const fs::path& path,
        const Instance& instance,
        std::uint64_t optimizer_seed,
        const SearchDiagnostics& diagnostics) {
    fs::create_directories(path.parent_path());
    const fs::path temporary = path.string() + ".tmp";
    std::ofstream output(temporary, std::ios::trunc);
    if (!output) throw std::runtime_error(
        "Cannot write convergence log: " + temporary.string());
    output << std::setprecision(17);
    output << "schema_version,condition_index,condition_id,instance_index,instance_seed,optimizer_seed,"
              "generation,search_evaluations,incumbent_total_tardiness,population_best,population_mean,"
              "population_median,population_worst,population_std,unique_chromosomes,stagnant_generations,"
              "last_improvement_generation,local_search_calls,local_search_evaluations,"
              "local_search_improvements,effective_mutation_probability,stop_reason,"
              "search_phase,baseline_generations,baseline_best_ttd,baseline_evaluations\n";
    for (const auto& row : diagnostics.convergence) {
        output << 1 << ',' << instance.condition_index << ','
               << csv_escape(instance.condition_id) << ',' << instance.instance_index << ','
               << instance.instance_seed << ',' << optimizer_seed << ',' << row.generation << ','
               << row.evaluations << ',' << row.incumbent_ttd << ',' << row.population_best << ','
               << row.population_mean << ',' << row.population_median << ',' << row.population_worst << ','
               << row.population_std << ',' << row.unique_chromosomes << ','
               << row.stagnant_generations << ',' << row.last_improvement_generation << ','
               << row.local_search_calls << ',' << row.local_search_evaluations << ','
               << row.local_search_improvements << ',' << row.effective_mutation_probability << ',';
        if (row.generation == diagnostics.convergence.back().generation) {
            output << diagnostics.stop_reason;
        }
        output << ',' << (row.refinement_phase ? "joint_neighborhood_refinement" : "memetic_search")
               << ',' << diagnostics.baseline_generations << ',' << diagnostics.baseline_best_ttd
               << ',' << diagnostics.baseline_evaluations << '\n';
    }
    output.close();
    atomic_replace(temporary, path);
}

void write_plan(
        const fs::path& path, const Instance& instance, std::uint64_t optimizer_seed,
        const Individual& best, std::uint64_t evaluations, int generations,
        double runtime_seconds, const std::string& dataset_contract_sha256,
        const std::string& optimizer_contract_sha256,
        const SearchDiagnostics& diagnostics,
        const fs::path& convergence_relative) {
    fs::create_directories(path.parent_path());
    const fs::path temporary = path.string() + ".tmp";
    std::ofstream output(temporary, std::ios::trunc);
    if (!output) throw std::runtime_error("Cannot write plan: " + temporary.string());
    output << std::setprecision(17);
    output << "schema_version,dataset_contract_sha256,optimizer_contract_sha256,condition_index,condition_id,"
              "instance_index,instance_seed,optimizer_seed,objective_total_tardiness,surrogate_makespan,"
              "total_completion_time,planned_maintenance_time,search_evaluations,generations,runtime_seconds,"
              "initial_objective_total_tardiness,improvement_percent,last_improvement_generation,stop_reason,"
              "convergence_file,final_unique_chromosomes,"
              "operation_index,sequence_index,machine_id,machine_class,machine_capacity,action,planned_start,"
              "planned_end,family,batch_size,job_ids,expected_health_before,expected_health_after,"
              "baseline_best_ttd,baseline_generations,baseline_evaluations\n";
    for (std::size_t index = 0; index < best.evaluation.operations.size(); ++index) {
        const auto& operation = best.evaluation.operations[index];
        output << 1 << ',' << dataset_contract_sha256 << ',' << optimizer_contract_sha256 << ','
               << instance.condition_index << ',' << csv_escape(instance.condition_id) << ','
               << instance.instance_index << ',' << instance.instance_seed << ',' << optimizer_seed << ','
               << best.evaluation.total_tardiness << ',' << best.evaluation.makespan << ','
               << best.evaluation.total_completion << ',' << best.evaluation.planned_maintenance_time << ','
               << evaluations << ',' << generations << ',' << runtime_seconds << ','
               << diagnostics.initial_best_ttd << ','
               << (diagnostics.initial_best_ttd > 0.0
                    ? 100.0 * (diagnostics.initial_best_ttd - best.evaluation.total_tardiness)
                        / diagnostics.initial_best_ttd
                    : 0.0) << ','
               << diagnostics.last_improvement_generation << ','
               << diagnostics.stop_reason << ',' << csv_escape(convergence_relative.generic_string()) << ','
               << diagnostics.convergence.back().unique_chromosomes << ',' << index << ','
               << operation.sequence_index << ',' << operation.machine << ',' << kMachineClass[operation.machine]
               << ',' << kMachineCapacity[operation.machine] << ',' << operation.action << ','
               << operation.start << ',' << operation.end << ',';
        if (operation.family >= 0) output << operation.family;
        output << ',' << operation.total_size << ',' << csv_escape(join_jobs(operation.jobs)) << ','
               << operation.health_before << ',' << operation.health_after << ','
               << diagnostics.baseline_best_ttd << ',' << diagnostics.baseline_generations
               << ',' << diagnostics.baseline_evaluations << '\n';
    }
    output.close();
    atomic_replace(temporary, path);
}

ResultSummary summarize(
        const Instance& instance, std::uint64_t optimizer_seed, const fs::path& relative_path,
        const Individual& best, std::uint64_t evaluations, int generations,
        double runtime_seconds, std::string status,
        const SearchDiagnostics& diagnostics,
        const fs::path& convergence_relative) {
    int batches = 0;
    int ipm = 0;
    int pm = 0;
    for (const auto& operation : best.evaluation.operations) {
        batches += operation.action == "PROCESS";
        ipm += operation.action == "IPM";
        pm += operation.action == "PM";
    }
    return ResultSummary{
        instance.condition_index, instance.condition_id, instance.instance_index,
        instance.instance_seed, optimizer_seed, relative_path.generic_string(),
        best.evaluation.total_tardiness, best.evaluation.makespan,
        best.evaluation.total_completion, best.evaluation.planned_maintenance_time,
        batches, ipm, pm, evaluations, generations,
        diagnostics.initial_best_ttd,
        diagnostics.initial_best_ttd > 0.0
            ? 100.0 * (diagnostics.initial_best_ttd - best.evaluation.total_tardiness)
                / diagnostics.initial_best_ttd
            : 0.0,
        diagnostics.last_improvement_generation,
        diagnostics.stop_reason,
        convergence_relative.generic_string(),
        diagnostics.convergence.back().unique_chromosomes,
        runtime_seconds,
        std::move(status),
        diagnostics.baseline_best_ttd,
        diagnostics.baseline_generations,
        diagnostics.baseline_evaluations
    };
}

std::optional<ResultSummary> read_existing_summary(
        const fs::path& plan_path, const fs::path& relative_path,
        const std::string& dataset_contract_sha256,
        const std::string& optimizer_contract_sha256) {
    std::ifstream input(plan_path);
    std::string header_line;
    std::string row_line;
    if (!input || !std::getline(input, header_line) || !std::getline(input, row_line)) return std::nullopt;
    const auto header = header_map(parse_csv_line(header_line));
    const auto row = parse_csv_line(row_line);
    ResultSummary result;
    try {
        result.condition_index = std::stoi(required_field(row, header, "condition_index"));
        result.condition_id = required_field(row, header, "condition_id");
        result.instance_index = std::stoi(required_field(row, header, "instance_index"));
        result.instance_seed = std::stoull(required_field(row, header, "instance_seed"));
        result.optimizer_seed = std::stoull(required_field(row, header, "optimizer_seed"));
        result.objective_ttd = std::stod(required_field(row, header, "objective_total_tardiness"));
        result.surrogate_makespan = std::stod(required_field(row, header, "surrogate_makespan"));
        result.total_completion = std::stod(required_field(row, header, "total_completion_time"));
        result.planned_maintenance_time = std::stod(required_field(row, header, "planned_maintenance_time"));
        result.evaluations = std::stoull(required_field(row, header, "search_evaluations"));
        result.generations = std::stoi(required_field(row, header, "generations"));
        result.initial_objective_ttd = std::stod(required_field(
            row, header, "initial_objective_total_tardiness"));
        result.improvement_percent = std::stod(required_field(row, header, "improvement_percent"));
        result.last_improvement_generation = std::stoi(required_field(
            row, header, "last_improvement_generation"));
        result.stop_reason = required_field(row, header, "stop_reason");
        result.convergence_file = required_field(row, header, "convergence_file");
        result.final_unique_chromosomes = static_cast<std::size_t>(std::stoull(
            required_field(row, header, "final_unique_chromosomes")));
        result.runtime_seconds = std::stod(required_field(row, header, "runtime_seconds"));
        result.plan_file = relative_path.generic_string();
        result.status = "resumed";
        result.baseline_best_ttd = std::stod(required_field(row, header, "baseline_best_ttd"));
        result.baseline_generations = std::stoi(required_field(row, header, "baseline_generations"));
        result.baseline_evaluations = std::stoull(required_field(row, header, "baseline_evaluations"));
        input.clear();
        input.seekg(0);
        std::getline(input, header_line);
        while (std::getline(input, row_line)) {
            if (row_line.empty()) continue;
            const auto values = parse_csv_line(row_line);
            const auto action = required_field(values, header, "action");
            result.batch_count += action == "PROCESS";
            result.ipm_count += action == "IPM";
            result.pm_count += action == "PM";
        }
    } catch (...) {
        return std::nullopt;
    }
    return result;
}

void write_manifest(const fs::path& path, std::vector<ResultSummary> results) {
    std::sort(results.begin(), results.end(), [](const ResultSummary& a, const ResultSummary& b) {
        return std::tie(a.condition_index, a.instance_index)
             < std::tie(b.condition_index, b.instance_index);
    });
    const fs::path temporary = path.string() + ".tmp";
    std::ofstream output(temporary, std::ios::trunc);
    if (!output) throw std::runtime_error("Cannot write plan manifest.");
    output << std::setprecision(17);
    output << "schema_version,condition_index,condition_id,instance_index,instance_seed,optimizer_seed,plan_file,"
              "objective_total_tardiness,surrogate_makespan,total_completion_time,planned_maintenance_time,"
              "batch_count,ipm_count,pm_count,search_evaluations,generations,initial_objective_total_tardiness,"
              "improvement_percent,last_improvement_generation,stop_reason,convergence_file,"
              "final_unique_chromosomes,runtime_seconds,status,baseline_best_ttd,"
              "baseline_generations,baseline_evaluations,refinement_improvement_percent\n";
    for (const auto& row : results) {
        output << 2 << ',' << row.condition_index << ',' << csv_escape(row.condition_id) << ','
               << row.instance_index << ',' << row.instance_seed << ',' << row.optimizer_seed << ','
               << csv_escape(row.plan_file) << ',' << row.objective_ttd << ',' << row.surrogate_makespan << ','
               << row.total_completion << ',' << row.planned_maintenance_time << ',' << row.batch_count << ','
               << row.ipm_count << ',' << row.pm_count << ',' << row.evaluations << ',' << row.generations << ','
               << row.initial_objective_ttd << ',' << row.improvement_percent << ','
               << row.last_improvement_generation << ',' << row.stop_reason << ','
               << csv_escape(row.convergence_file) << ',' << row.final_unique_chromosomes << ','
               << row.runtime_seconds << ',' << row.status << ',' << row.baseline_best_ttd
               << ',' << row.baseline_generations << ',' << row.baseline_evaluations
               << ',' << (row.baseline_best_ttd > 0
                   ? 100.0 * (row.baseline_best_ttd - row.objective_ttd) / row.baseline_best_ttd : 0.0) << '\n';
    }
    output.close();
    atomic_replace(temporary, path);
}

struct Arguments {
    fs::path dataset_dir;
    fs::path manifest;
    fs::path output_dir;
    std::string dataset_contract_sha256;
    std::string optimizer_contract_sha256;
    Parameters parameters;
};

void print_help() {
    std::cout << "Offline memetic planner for frozen E1 batch-scheduling instances\n\n"
              << "Required:\n"
              << "  --dataset-dir PATH --manifest PATH --output-dir PATH\n"
              << "  --dataset-contract-sha256 HEX --optimizer-contract-sha256 HEX\n"
              << "Optional search parameters mirror the Python launcher.\n";
}

Arguments parse_arguments(int argc, char** argv) {
    Arguments args;
    auto require_value = [&](int& index) -> std::string {
        if (++index >= argc) throw std::runtime_error("Missing command-line value.");
        return argv[index];
    };
    for (int i = 1; i < argc; ++i) {
        const std::string name = argv[i];
        if (name == "--help" || name == "-h") { print_help(); std::exit(0); }
        else if (name == "--dataset-dir") args.dataset_dir = require_value(i);
        else if (name == "--manifest") args.manifest = require_value(i);
        else if (name == "--output-dir") args.output_dir = require_value(i);
        else if (name == "--dataset-contract-sha256") args.dataset_contract_sha256 = require_value(i);
        else if (name == "--optimizer-contract-sha256") args.optimizer_contract_sha256 = require_value(i);
        else if (name == "--population") args.parameters.population = std::stoi(require_value(i));
        else if (name == "--search-strategy") args.parameters.search_strategy = require_value(i);
        else if (name == "--refinement-generations") args.parameters.refinement_generations = std::stoi(require_value(i));
        else if (name == "--generations") args.parameters.generations = std::stoi(require_value(i));
        else if (name == "--local-search-individuals") args.parameters.local_search_individuals = std::stoi(require_value(i));
        else if (name == "--local-search-evaluations") args.parameters.local_search_evaluations = std::stoi(require_value(i));
        else if (name == "--nonimproving-generations") args.parameters.nonimproving_generations = std::stoi(require_value(i));
        else if (name == "--local-search-interval") args.parameters.local_search_interval = std::stoi(require_value(i));
        else if (name == "--mutation-adaptation-generations") args.parameters.mutation_adaptation_generations = std::stoi(require_value(i));
        else if (name == "--mutation-probability") args.parameters.mutation_probability = std::stod(require_value(i));
        else if (name == "--formation-probability-ga") args.parameters.formation_probability_ga = std::stod(require_value(i));
        else if (name == "--formation-probability-ls") args.parameters.formation_probability_ls = std::stod(require_value(i));
        else if (name == "--local-search-deviation-percent") args.parameters.local_search_deviation_percent = std::stod(require_value(i));
        else if (name == "--arrival-window-factor") args.parameters.arrival_window_factor = std::stod(require_value(i));
        else if (name == "--planning-seed-base") args.parameters.planning_seed_base = std::stoull(require_value(i));
        else if (name == "--threads") args.parameters.threads = std::stoi(require_value(i));
        else if (name == "--instance-limit") args.parameters.instance_limit = std::stoi(require_value(i));
        else if (name == "--no-resume") args.parameters.resume = false;
        else throw std::runtime_error("Unknown command-line argument: " + name);
    }
    if (args.dataset_dir.empty() || args.manifest.empty() || args.output_dir.empty()
            || args.dataset_contract_sha256.empty() || args.optimizer_contract_sha256.empty()) {
        throw std::runtime_error("Required command-line arguments are missing. Use --help.");
    }
    const auto& p = args.parameters;
    if ((p.search_strategy != "legacy" && p.search_strategy != "joint-neighborhood")
            || p.refinement_generations < 0
            || p.population < 4 || p.population % 2 || p.generations <= 0
            || p.local_search_individuals < 0 || p.local_search_evaluations < 0
            || p.nonimproving_generations <= 0 || p.local_search_interval <= 0
            || p.mutation_adaptation_generations <= 0
            || p.threads <= 0
            || p.mutation_probability < 0 || p.mutation_probability > 1
            || p.formation_probability_ga < 0 || p.formation_probability_ga > 1
            || p.formation_probability_ls < 0 || p.formation_probability_ls > 1
            || p.local_search_deviation_percent < 0
            || p.arrival_window_factor <= 0) {
        throw std::runtime_error("Invalid optimizer parameter.");
    }
    return args;
}

int run(const Arguments& args) {
    auto instances = read_instance_manifest(args.dataset_dir, args.manifest);
    if (args.parameters.instance_limit > 0
            && args.parameters.instance_limit < static_cast<int>(instances.size())) {
        instances.resize(static_cast<std::size_t>(args.parameters.instance_limit));
    }
    if (instances.empty()) throw std::runtime_error("No structural instances selected.");
    fs::create_directories(args.output_dir / "plans");
    fs::create_directories(args.output_dir / "convergence");
    std::cout << "[offline-ma] structural_instances=" << instances.size()
              << " search_strategy=" << args.parameters.search_strategy
              << " population=" << args.parameters.population
              << " generations=" << args.parameters.generations
              << " refinement_generations=" << (args.parameters.search_strategy == "legacy"
                  ? 0 : args.parameters.refinement_generations)
              << " threads=" << args.parameters.threads << std::endl;
    std::vector<ResultSummary> results(instances.size());
    std::atomic<std::size_t> next{0};
    std::atomic<std::size_t> completed{0};
    std::atomic<bool> failed{false};
    std::mutex output_mutex;
    std::mutex error_mutex;
    std::string first_error;
    const int worker_count = std::min<int>(args.parameters.threads, static_cast<int>(instances.size()));
    auto worker = [&]() {
        while (!failed.load()) {
            const std::size_t index = next.fetch_add(1);
            if (index >= instances.size()) return;
            try {
                auto& instance = instances[index];
                const fs::path relative = fs::path("plans") / instance.condition_id
                    / ("instance-" + [&] { std::ostringstream s; s << std::setw(3) << std::setfill('0') << instance.instance_index; return s.str(); }() + ".csv");
                const fs::path target = args.output_dir / relative;
                const fs::path convergence_relative = fs::path("convergence")
                    / instance.condition_id
                    / ("instance-" + [&] { std::ostringstream s; s << std::setw(3)
                        << std::setfill('0') << instance.instance_index; return s.str(); }() + ".csv");
                const fs::path convergence_target = args.output_dir / convergence_relative;
                if (args.parameters.resume && fs::exists(target)) {
                    auto existing = read_existing_summary(
                        target, relative, args.dataset_contract_sha256,
                        args.optimizer_contract_sha256);
                    if (!existing) throw std::runtime_error("Existing plan is malformed: " + target.string());
                    if (!fs::exists(args.output_dir / existing->convergence_file)) {
                        throw std::runtime_error(
                            "Existing plan is missing its convergence log: "
                            + (args.output_dir / existing->convergence_file).string());
                    }
                    results[index] = *existing;
                } else {
                    load_jobs(instance);
                    const std::uint64_t optimizer_seed = args.parameters.planning_seed_base
                        + static_cast<std::uint64_t>(instance.condition_index) * 1000ULL
                        + static_cast<std::uint64_t>(instance.instance_index);
                    const auto started = std::chrono::steady_clock::now();
                    Optimizer optimizer(instance, args.parameters, optimizer_seed);
                    int generations = 0;
                    auto best = optimizer.run(generations);
                    const double seconds = std::chrono::duration<double>(
                        std::chrono::steady_clock::now() - started).count();
                    write_convergence(
                        convergence_target, instance, optimizer_seed,
                        optimizer.diagnostics());
                    write_plan(target, instance, optimizer_seed, best, optimizer.evaluations(),
                               generations, seconds, args.dataset_contract_sha256,
                               args.optimizer_contract_sha256,
                               optimizer.diagnostics(), convergence_relative);
                    results[index] = summarize(instance, optimizer_seed, relative, best,
                                               optimizer.evaluations(), generations, seconds, "generated",
                                               optimizer.diagnostics(), convergence_relative);
                }
                const std::size_t done = completed.fetch_add(1) + 1;
                std::lock_guard<std::mutex> lock(output_mutex);
                std::cout << "[offline-ma] " << done << '/' << instances.size()
                          << " condition=" << instance.condition_id
                          << " instance=" << instance.instance_index
                          << " TTD=" << results[index].objective_ttd
                          << " baseline_TTD=" << results[index].baseline_best_ttd
                          << " batches=" << results[index].batch_count
                          << " IPM=" << results[index].ipm_count
                          << " PM=" << results[index].pm_count
                          << " generations=" << results[index].generations
                          << " last_improvement=" << results[index].last_improvement_generation
                          << " improvement=" << results[index].improvement_percent << '%'
                          << " unique=" << results[index].final_unique_chromosomes
                          << " stop=" << results[index].stop_reason
                          << " status=" << results[index].status << std::endl;
            } catch (const std::exception& error) {
                failed.store(true);
                std::lock_guard<std::mutex> lock(error_mutex);
                if (first_error.empty()) first_error = error.what();
                return;
            }
        }
    };
    std::vector<std::thread> workers;
    for (int i = 0; i < worker_count; ++i) workers.emplace_back(worker);
    for (auto& thread : workers) thread.join();
    if (failed.load()) throw std::runtime_error(first_error);
    write_manifest(args.output_dir / "plan_manifest.csv", std::move(results));
    std::cout << "[offline-ma] complete plan_manifest="
              << (args.output_dir / "plan_manifest.csv").string() << std::endl;
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        return run(parse_arguments(argc, argv));
    } catch (const std::exception& error) {
        std::cerr << "[offline-ma] ERROR: " << error.what() << std::endl;
        return 2;
    }
}
