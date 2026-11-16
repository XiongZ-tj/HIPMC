"""Causal snapshot boundary and persistent C++ worker."""
from pathlib import Path
import subprocess
import shutil
import time

from methods.baselines.sdr.rules import proposal_from_batch

ROOT = Path(__file__).resolve().parents[2]


def build(build_dir):
    directory = Path(build_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / 'rh_alns.exe'
    source = Path(__file__).with_name('planner.cpp')
    compiler = shutil.which('g++') or shutil.which('clang++')
    if not compiler:
        raise RuntimeError('A C++20 compiler must be installed in PATH')
    subprocess.run([compiler, '-std=c++20', '-O3', '-DNDEBUG', '-pthread', str(source), '-o', str(target)], check=True)
    return target


class Controller:
    control_trajectory_scope = 'physical'

    def __init__(self, executable, budget=8000, worker_arguments=()):
        if budget < 1:
            raise ValueError('budget must be positive')
        self.process = subprocess.Popen([str(executable),*worker_arguments], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1)
        self.budget = budget
        self.trace = []
        self.plan = []
        self.key = None

    def close(self):
        if self.process.stdin:
            self.process.stdin.close()
        self.process.wait(timeout=10)
        self.process.stdout.close()

    def begin_episode(self, seed):
        self.seed = int(seed)
        self.trace = []
        self.plan = []
        self.key = None
        self.process.stdin.write('RESET\n'); self.process.stdin.flush()
        if self.process.stdout.readline().strip() != 'OK':
            raise RuntimeError('Worker reset failed')

    def bind_episode(self, **kwargs):
        # The shared simulator calls this with its full job list. Deliberately
        # retain nothing: planning receives only the filtered snapshot below.
        pass

    def observe(self, transition):
        pass

    def update(self, snapshot, requester_id=-1):
        clock=snapshot['clock']
        key = (clock,requester_id,tuple(snapshot['machines']),tuple(snapshot['jobs']))
        if key == self.key:
            return
        self.key = key
        machines, jobs = snapshot['machines'], snapshot['jobs']
        lines = [f"SOLVE {clock:.17g} {snapshot['multiplier']} {self.budget} {self.seed+len(self.trace)} {len(jobs)} {len(machines)} {requester_id}"]
        lines.extend(' '.join(map(str, row)) for row in machines)
        lines.extend(' '.join(map(str, row)) for row in jobs)
        started = time.perf_counter()
        self.process.stdin.write('\n'.join(lines)+'\n'); self.process.stdin.flush()
        header = self.process.stdout.readline().split()
        if not header or header[0] != 'RESULT':
            raise RuntimeError('RH-ALNS worker failed')
        self.plan = []
        for _ in range(int(header[5])):
            row = list(map(int, self.process.stdout.readline().split()))
            if len(row) != row[2]+3:
                raise RuntimeError('Malformed worker response')
            self.plan.append((row[0], row[1], row[3:]))
        self.trace.append(dict(clock=clock, requester_id=requester_id, jobs=len(jobs), evaluations=int(header[1]),
                               iterations=int(header[2]), initial_ttd=float(header[3]),
                               predicted_ttd=float(header[4]), seconds=time.perf_counter()-started,
                               work_conserving_reassignments=0,masked_action_corrections=0,
                               planned_machine='',planned_action='',planned_job_ids='',
                               submitted_action='',submitted_job_ids=''))

    def build_proposals(self, jobs, capacity, clock, processing_times, *, requester_id, **kwargs):
        available = {job.job_id: job for job in jobs}
        choices = [b for b in self.plan if all(j in available for j in b[2]) and sum(available[j].size for j in b[2]) <= capacity]
        assigned = [b for b in choices if b[0] == requester_id]
        if not assigned:
            return []
        chosen = assigned[0]
        self.action = chosen[1]
        self.selected = chosen
        self.trace[-1].update(planned_machine=requester_id,planned_action=self.action,
                             planned_job_ids=';'.join(map(str,chosen[2])))
        family = available[chosen[2][0]].family
        return [proposal_from_batch((family, tuple(chosen[2])), jobs, capacity, clock, processing_times, 'OPT')]

    def select_action(self, observation, *, retained_batch=False, **kwargs):
        if retained_batch:
            return 0
        action = self.action
        self.trace[-1].update(submitted_action=action,
                             submitted_job_ids=';'.join(map(str,self.selected[2])))
        if action == 0:
            self.plan.remove(self.selected)
        return action
