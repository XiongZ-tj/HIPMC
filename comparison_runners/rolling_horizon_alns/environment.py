"""Only decision routing is specialized; physical event handlers are inherited."""
from methods.hierarchical.environment import HierarchicalSchedulingEnv
from methods.hierarchical.types import MachinePhase
import heapq
import torch


class Environment(HierarchicalSchedulingEnv):
    def _advance_with_repairs(self):
        super()._advance_with_repairs()
        # Drain simultaneous physical events before the next planning boundary.
        while self._event_queue and self._event_queue[0][0] <= self.clock:
            at, _, machine_id, event, generation = heapq.heappop(self._event_queue)
            machine = self.machines[machine_id]
            if generation != machine.generation:
                continue
            if event == 'repair_complete':
                self._handle_repair_completion(machine)
            else:
                self._handle_machine_event(machine, event, at)

    def _snapshot(self):
        jobs = [(j.job_id, j.family, j.size, j.due_time, self.job_cluster_pt[j.family])
                for j in self._available_jobs()]
        machines = []
        for m in self.machines:
            health=min(3,m.state)
            # Never expose scheduled_at, next-arrival times, or sampled outcomes.
            remaining = 0.
            if m.phase == MachinePhase.PROCESSING:
                elapsed = max(0., self.clock-m.last_event_at)
                remaining = max(0., m.remaining_work-elapsed*(1-m.state/30))/(1-m.state/30)
            elif m.phase == MachinePhase.MAINTAINING:
                duration = (self.config.ipm_times if m.maintenance_action == 1 else self.config.pm_times)[m.class_id]
                remaining = max(0., duration-(self.clock-m.activity_started_at))
                if m.maintenance_action == 2:
                    health=0
                else:
                    q=torch.zeros((4,4),dtype=torch.float64)
                    for state,rate in enumerate(self.config.ipm_recovery_rates[m.class_id],start=1):
                        q[state,state-1]=rate;q[state,state]=-rate
                    probabilities=torch.matrix_exp(q*duration)[m.state]
                    health=float(probabilities@torch.arange(4,dtype=torch.float64))
            elif m.phase == MachinePhase.REPAIRING:
                failures=[r['time'] for r in m.history if r.get('event')=='failure']
                elapsed = max(0., self.clock-max(failures,default=self.clock))
                if m.failure_type == -1:
                    remaining = max(0., self.config.minimal_repair_times[m.class_id]-elapsed)
                else:
                    lo, hi = self.config.corrective_time_ranges[m.class_id]
                    duration = (lo, (lo+hi)/2, hi)[m.failure_type]
                    remaining = max(0., duration-elapsed)
            if m.batch_ids and m.phase != MachinePhase.PROCESSING:
                remaining += m.remaining_work/(1-m.state/30)
            machines.append((m.machine_id, m.class_id, m.capacity,
                             int(m.phase == MachinePhase.IDLE and not m.batch_ids),
                             self.clock+remaining, health))
        return dict(clock=self.clock, multiplier=self.config.degradation_multiplier, machines=machines, jobs=jobs)

    def _request(self, machine, *, epsilon, training):
        if training:
            raise ValueError('RH-ALNS is an optimization baseline, not a learned policy')
        if not machine.batch_ids:
            self.controller.update(self._snapshot(),requester_id=machine.machine_id)
        return super()._request(machine, epsilon=epsilon, training=False)
