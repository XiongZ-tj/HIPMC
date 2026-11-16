// Stateful stdin/stdout worker. Only observed snapshots cross this boundary.
#define main unused_offline_cli
#include "../offline_memetic_ma/cpp/offline_memetic_planner.cpp"
#undef main

namespace rh {
struct Task { int id, family, size; double due, work; };
struct Resource { int id, cls, capacity, idle; double ready, health; };
struct Batch { int machine, action; std::vector<int> ids; };
using Plan = std::vector<Batch>;
struct Search {
    std::vector<Task> jobs;
    std::vector<Resource> machines;
    std::mt19937_64 rng;
    double now{}, multiplier{1};
    int evaluations{}, budget{}, requester{-1};
    bool enhanced{true};
    double recovered_health(int cls, double health) const {
        // Cache integer-state integrations for the worker's fixed physical parameters.
        static const auto recovery = [] {
            std::array<std::array<double,4>,3> table{};
            for(int c=0;c<3;++c)for(int k=0;k<4;++k)
                table[c][k]=expected_ipm_state_integer(c,k);
            return table;
        }();
        health=std::clamp(health,0.0,3.0);
        int lower=static_cast<int>(std::floor(health));
        int upper=static_cast<int>(std::ceil(health));
        if(lower==upper)return recovery[cls][lower];
        double fraction=health-lower;
        return recovery[cls][lower]*(1.0-fraction)+recovery[cls][upper]*fraction;
    }
    std::map<std::tuple<int,double,double>,std::pair<double,double>> duration_cache;
    void decode_prefix(Plan& plan) {
        if(requester<0 || !machines[requester].idle)return;
        auto head=std::find_if(plan.begin(),plan.end(),[&](const Batch& b){return b.machine==requester;});
        if(head==plan.end()) {
            head=std::find_if(plan.begin(),plan.end(),[&](const Batch& b){
                int size=0;for(int j:b.ids)size+=jobs[j].size;
                return size<=machines[requester].capacity;
            });
            if(head!=plan.end())head->machine=requester;
        }
        // Healthy-machine maintenance is outside the common action space.
        // Enforce this in every evaluated candidate, not after optimization.
        if(head!=plan.end() && machines[requester].health==0)head->action=0;
    }
    double duration(int cls, double health, double work, double& final_health) {
        const auto key=std::tuple(cls,health,work);
        auto cached=duration_cache.find(key);
        if(cached!=duration_cache.end()) {final_health=cached->second.second;return cached->second.first;}
        // Same mean-state surrogate as offline MA, extended to residual work.
        double elapsed=0;
        const auto [dl,dh]=kDegradationRange[cls];
        const auto [fl,fh]=kFailureRange[cls];
        std::array<double,4> degradation{dl,(dl+dh)/2,dh,kTerminalDegradationRate[cls]};
        std::array<double,4> failures{fl,fl+(fh-fl)/3,fl+2*(fh-fl)/3,fh};
        while(work>1e-10) {
            double rate=1-health/30, dt=std::min(1.,work/rate);
            work-=rate*dt;
            double failure=multiplier*lerp_state(failures,health);
            double repair=.6*kCorrectiveRange[cls][0]+.3*(kCorrectiveRange[cls][0]+kCorrectiveRange[cls][1])/2+.1*kCorrectiveRange[cls][1];
            elapsed+=dt*(1+failure*repair);
            double d=multiplier*lerp_state(degradation,health);
            if(health>=3-1e-12) elapsed+=dt*d*kMinimalRepairTime[cls];
            else health=std::min(3.,health+d*dt);
        }
        final_health=health;
        if(duration_cache.size()>=65536)duration_cache.clear();
        duration_cache.emplace(key,std::pair(elapsed,health));
        return elapsed;
    }
    double score(Plan& plan) {
        decode_prefix(plan);
        ++evaluations;
        double total=0;
        std::vector<double> time,health;
        for(auto m:machines){time.push_back(m.ready);health.push_back(m.health);}
        for(const auto& b:plan) {
            if(b.ids.empty()) continue;
            int m=b.machine,c=machines[m].cls;
            if(b.action==1){time[m]+=kIpmTime[c];health[m]=recovered_health(c,health[m]);}
            if(b.action==2){time[m]+=kPmTime[c];health[m]=0;}
            time[m]+=duration(c,health[m],jobs[b.ids[0]].work,health[m]);
            for(int id:b.ids) total+=std::max(0.,time[m]-jobs[id].due);
        }
        return total;
    }
    double insert(Plan& plan,int id,bool search,int allowance=1000000000) {
        if(!search) {
            for(auto& b:plan) {
                int size=jobs[id].size;for(int j:b.ids)size+=jobs[j].size;
                if(jobs[b.ids[0]].family==jobs[id].family && size<=machines[b.machine].capacity){b.ids.push_back(id);return 0;}
            }
            int selected=-1;double ready=1e100;
            for(int m=0;m<(int)machines.size();++m)if(machines[m].capacity>=jobs[id].size && machines[m].ready<ready){selected=m;ready=machines[m].ready;}
            if(selected<0)throw std::runtime_error("infeasible job");
            plan.push_back(Batch{selected,0,{id}});return 0;
        }
        int start=evaluations;
        Plan best; double value=std::numeric_limits<double>::infinity(), second=value;
        auto consider=[&](Plan candidate) {
            if(search && (evaluations>=budget || evaluations-start>=allowance)) return;
            double cost=score(candidate);
            if(cost<value){second=value;value=cost;best=std::move(candidate);}
            else second=std::min(second,cost);
        };
        // Always construct a feasible fallback, even when the search budget expires.
        int machine=0;
        while(machine<(int)machines.size() && machines[machine].capacity<jobs[id].size) ++machine;
        if(machine==(int)machines.size()) throw std::runtime_error("infeasible job size");
        best=plan; best.push_back(Batch{machine,0,{id}});
        if(!search || evaluations<budget){value=score(best);}
        for(int i=0;i<(int)plan.size();++i) {
            if(evaluations>=budget || evaluations-start>=allowance) {
                plan=std::move(best);
                return std::isfinite(second)?second-value:0;
            }
            const auto& b=plan[i]; int size=jobs[id].size;
            for(int j:b.ids)size+=jobs[j].size;
            if(jobs[b.ids[0]].family==jobs[id].family && size<=machines[b.machine].capacity){
                Plan candidate=plan;candidate[i].ids.push_back(id);consider(std::move(candidate));
            }
        }
        for(int m=0;m<(int)machines.size();++m) if(jobs[id].size<=machines[m].capacity)
            for(int i=0;i<=(int)plan.size();++i) for(int a=0;a<3;++a){
                if(evaluations>=budget || evaluations-start>=allowance) {
                    plan=std::move(best);
                    return std::isfinite(second)?second-value:0;
                }
                Plan candidate=plan;candidate.insert(candidate.begin()+i,Batch{m,a,{id}});consider(std::move(candidate));
            }
        plan=std::move(best);
        return std::isfinite(second)?second-value:0;
    }
    int load(const Batch& b) const {int value=0;for(int j:b.ids)value+=jobs[j].size;return value;}
    void joint_descent(Plan& plan,int allowance) {
        const int end=std::min(budget-1,evaluations+allowance);
        if(evaluations>=end)return;
        double value=score(plan);Plan origin=plan;
        std::vector<int> batches(origin.size()),resources(machines.size());
        std::iota(batches.begin(),batches.end(),0);std::iota(resources.begin(),resources.end(),0);
        std::shuffle(batches.begin(),batches.end(),rng);std::shuffle(resources.begin(),resources.end(),rng);
        for(int i:batches)for(int m:resources)if(load(origin[i])<=machines[m].capacity)
          for(int position=0;position<(int)origin.size();++position)for(int action=0;action<3;++action){
            if(evaluations>=end)return;
            auto candidate=origin;auto moving=candidate[i];moving.machine=m;moving.action=action;
            candidate.erase(candidate.begin()+i);candidate.insert(candidate.begin()+position,moving);
            double next=score(candidate);if(next<value){value=next;plan=std::move(candidate);}
          }
    }
    Plan inherit_batches(const Plan& receiver,const Plan& donor){
        if(receiver.empty())return donor;
        int left=rng()%receiver.size(),right=left+rng()%(receiver.size()-left);
        std::vector<bool> used(jobs.size());Plan result;
        for(int i=left;i<=right;++i){result.push_back(receiver[i]);for(int j:receiver[i].ids)used[j]=true;}
        for(auto b:donor){std::erase_if(b.ids,[&](int j){return used[j];});if(!b.ids.empty())result.push_back(b);}
        return result;
    }
    Plan solve(const Plan& warm,int& iterations,double& initial,double& final) {
        Plan current;std::vector<bool> used(jobs.size());
        // Warm plan stores external IDs. Missing/completed jobs are removed.
        for(auto b:warm){std::vector<int> ids;
            for(int external:b.ids)for(int j=0;j<(int)jobs.size();++j)
                if(jobs[j].id==external && !used[j]){ids.push_back(j);used[j]=true;}
            if(!ids.empty() && b.machine<(int)machines.size()) {b.ids=ids;current.push_back(b);}
        }
        std::vector<int> missing;
        for(int j=0;j<(int)jobs.size();++j)if(!used[j])missing.push_back(j);
        std::sort(missing.begin(),missing.end(),[&](int a,int b){return jobs[a].due<jobs[b].due;});
        for(int j:missing)insert(current,j,false);
        // Initial construction costs count toward the complete optimization cost.
        initial=score(current);double cost=initial,best_cost=cost;
        Plan best=current;
        std::vector<double> weights(enhanced?8:4,1.);
        std::vector<std::pair<double,Plan>> archive{{cost,current}};
        double temperature=std::max(1.,initial*.05);
        iterations=0;
        while(evaluations<budget && !jobs.empty()) {
            ++iterations;
            std::discrete_distribution<int> select(weights.begin(),weights.end());int op=select(rng);
            std::vector<int> removed;
            Plan candidate=current;
            int count=std::min(3,std::max(1,(int)jobs.size()/5));
            std::vector<int> order(jobs.size());std::iota(order.begin(),order.end(),0);
            std::shuffle(order.begin(),order.end(),rng);
            if(op==1)std::stable_sort(order.begin(),order.end(),[&](int a,int b){return jobs[a].due<jobs[b].due;});
            if(op==2){int family=jobs[order[0]].family;std::stable_sort(order.begin(),order.end(),[&](int a,int b){return (jobs[a].family==family)>(jobs[b].family==family);});}
            if(op==3 && !candidate.empty()){
                auto& b=candidate[rng()%candidate.size()];b.action=rng()%3;
            } else if(op==4) {
                joint_descent(candidate,256);
            } else if(op==7) {
                candidate=inherit_batches(candidate,archive[rng()%archive.size()].second);
                joint_descent(candidate,128);
            } else {
                if(op==5){
                    // Remove related batches before repair.
                    int family=jobs[order[0]].family;removed.clear();
                    for(auto& b:candidate)if(jobs[b.ids[0]].family==family)
                        removed.insert(removed.end(),b.ids.begin(),b.ids.end());
                }else if(op==6){
                    // Rebuild a machine queue fragment, preserving other queues.
                    int m=candidate[rng()%candidate.size()].machine;removed.clear();int batches=0;
                    for(auto& b:candidate)if(b.machine==m && batches++<2)
                        removed.insert(removed.end(),b.ids.begin(),b.ids.end());
                }else{
                removed.assign(order.begin(),order.begin()+std::min(count,(int)order.size()));
                }
                for(auto& b:candidate)std::erase_if(b.ids,[&](int j){return std::find(removed.begin(),removed.end(),j)!=removed.end();});
                std::erase_if(candidate,[](const Batch& b){return b.ids.empty();});
                if(iterations%2==0 && removed.size()<=3) {
                    // Regret-2 repair over feasible batch/machine/maintenance placements.
                    while(!removed.empty()) {
                        int choice=0;double greatest=-1;Plan selected;
                        const int allowance=128;
                        for(int i=0;i<(int)removed.size();++i){
                            Plan trial=candidate;
                            double regret=insert(trial,removed[i],true,allowance);
                            if(regret>greatest){greatest=regret;choice=i;selected=std::move(trial);}
                        }
                        candidate=std::move(selected);removed.erase(removed.begin()+choice);
                    }
                } else for(int j:removed)insert(candidate,j,true,128);
            }
            if(enhanced && iterations%8==0 && evaluations<budget-1)joint_descent(candidate,128);
            if(evaluations>=budget)break; // Never accept an incompletely optimized trial.
            double next=score(candidate);double reward=0;
            if(next<best_cost){best=candidate;best_cost=next;reward=5;}
            if(next<cost || std::generate_canonical<double,53>(rng)<std::exp((cost-next)/temperature)){
                reward=std::max(reward,next<cost?2.:.5);current=std::move(candidate);cost=next;
            }
            weights[op]=std::max(.05,.8*weights[op]+.2*reward);temperature*=.995;
            if(enhanced && (reward>=2 || iterations%20==0)){
                // Bounded elite archive, distinct objective levels avoid clones.
                bool duplicate=false;for(auto& entry:archive)if(std::abs(entry.first-next)<1e-9)duplicate=true;
                if(!duplicate){archive.emplace_back(next,candidate.empty()?current:candidate);
                    std::sort(archive.begin(),archive.end(),[](auto&a,auto&b){return a.first<b.first;});
                    if(archive.size()>8)archive.resize(8);
                }
            }
        }
        final=best_cost;
        for(auto& b:best)for(auto& id:b.ids)id=jobs[id].id;
        return best;
    }
};
}
#ifndef RH_ALNS_LIBRARY
int main(int argc,char** argv){
    std::ios::sync_with_stdio(false);
    std::vector<rh::Batch> warm;
    std::string command;
    while(std::cin>>command){
        if(command=="RESET"){warm.clear();std::cout<<"OK"<<std::endl;continue;}
        if(command!="SOLVE")return 2;
        rh::Search s;int n,m;unsigned long long seed;
        s.enhanced=!(argc>1 && std::string(argv[1])=="--prior-search");
        std::cin>>s.now>>s.multiplier>>s.budget>>seed>>n>>m>>s.requester;s.rng.seed(seed);
        for(int i=0;i<m;++i){rh::Resource r;std::cin>>r.id>>r.cls>>r.capacity>>r.idle>>r.ready>>r.health;s.machines.push_back(r);}
        for(int i=0;i<n;++i){rh::Task j;std::cin>>j.id>>j.family>>j.size>>j.due>>j.work;s.jobs.push_back(j);}
        int iterations;double initial,final;warm=s.solve(warm,iterations,initial,final);
        std::cout<<std::setprecision(17)<<"RESULT "<<s.evaluations<<' '<<iterations<<' '<<initial<<' '<<final<<' '<<warm.size()<<'\n';
        for(const auto& b:warm){std::cout<<b.machine<<' '<<b.action<<' '<<b.ids.size();for(int id:b.ids)std::cout<<' '<<id;std::cout<<'\n';}
        std::cout.flush();
    }
}
#endif
