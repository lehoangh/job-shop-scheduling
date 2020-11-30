#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 13:17:14 2020

@author: hoale
"""

"""
Single-stage scheduling problem with dissimilar parallel machines and sequence-
independent processing time.
1. Data from the supplementary resource of the paper: "Decomposition techniques
for multistage scheduling problems using mixed-integer and constraint programming
methods" 
(https://www.sciencedirect.com/science/article/pii/S009813540200100X)
"""

"""
This file is the hybrid classical approaches: MILP and MILP
"""

import os
import numpy as np
import gurobipy as grb
from read_data import read_data
# from hybrid.hybrid_relaxed_milp import _create_milp_model
# from hybrid.hybrid_milp_solver import _create_model
from gantt_plot import gantt_chart_plot, formulate_jobs_dict
import matplotlib.pyplot as plt
import dwavebinarycsp
import pyqubo
import dimod
from dwave.system import DWaveSampler, EmbeddingComposite


global maxOptTimsSeconds
maxOptTimsSeconds = -1


def callbackTerminate(model, where):
    global maxOptTimsSeconds
    if where != grb.GRB.Callback.POLLING:
        t = model.cbGet(grb.GRB.Callback.RUNTIME)
        if t >= maxOptTimsSeconds and maxOptTimsSeconds > 0:
            model.terminate()


""" Creation of relaxed MILP model with constraints (subproblem 1) """
def _create_milp_model(job_num, machine_num, job_ids, r_times, d_times, p_intervals, p_cost):
    """ Prepare the index for decision variables """
    # index of jobs
    jobs = tuple(job_ids)
    # index of machines
    machines = tuple(range(machine_num))
    # assignment of jobs on machines
    job_machine_pairs = [(i, m) for i in jobs for m in machines]

    """ Parameters model (dictionary) """
    # 1. release time
    release_time = dict(zip(jobs, tuple(r_times)))
    # 2. due time
    due_time = dict(zip(jobs, tuple(d_times)))
    # 3. processing time
    process_time = dict(zip(jobs, tuple(p_intervals)))
    # 4. processing cost
    job_cost = dict(zip(jobs, tuple(p_cost)))

    """ Create model """
    model = grb.Model("SSJSP")
    output_file = "gurobi-jss.log" + "_" + str(job_num) + "_" + str(machine_num)
    model.setParam(grb.GRB.Param.LogFile, output_file)

    """ Create decision variables """
    # 1. Assignments of jobs on machines
    x = model.addVars(job_machine_pairs, vtype=grb.GRB.BINARY, name="assign")
    # 2. Start time of executing each job (ts = time_start)
    ts = model.addVars(jobs, lb=0, name="start_time")

    """ Create the objective function """
    model.setObjective(grb.quicksum([(job_cost[i][m]*x[(i,m)]) for m in machines for i in jobs]),
                       sense=grb.GRB.MINIMIZE)

    """ Create constraints """
    # 1. job release time constraint
    model.addConstrs((ts[i] >= release_time[i] for i in jobs), name="job release constraint")
    # 2. job due time constraint
    model.addConstrs((ts[i] <= due_time[i] - grb.quicksum([process_time[i][m]*x[(i,m)] for m in machines])
                      for i in jobs), name="job due constraint")
    # 3. one job is assigned to one and only one machine
    model.addConstrs((grb.quicksum([x[(i,m)] for m in machines]) == 1 for i in jobs),
                     name="job non-splitting constraint")
    # # 4. A valid cut, it tightens LP relaxation, total processing time of all jobs assigned to same machine
    # # less than diff of latest due date & earliest release date
    # model.addConstrs((grb.quicksum([x[(i,m)]*process_time[i][m] for i in jobs]) <= max(due_time.values()) -
    #                   min(release_time.values()) for m in machines), name="total processing time of all jobs")

    return model, x, ts


""" Single-stage job shop scheduling problem by hybrid method of Gurobi MILP 
and Docplex CP """
class SSJSP_Hybrid(object):
    def __init__(self):
        # setting the solver attributes
        self.schedules = {}
        self.sequence = {}

    """ Solve the model and formulate the result """
    def solve(self, file_name, all_jobs, job_num, all_machines, machine_num,
              job_ids, request_times, due_times, process_intervals,
              processing_cost):
        solved = False
        k = 0
        milp_model, assign, start_time = _create_milp_model(job_num, machine_num, job_ids, request_times,
                                                            due_times, process_intervals, processing_cost)

        """ Write a log file, cannot call in main() function """
        output_file = os.getcwd() + "/logs/hybrid/2nd/hybrid-jss-" + file_name + ".log"
        milp_model.setParam(grb.GRB.Param.LogFile, output_file)

        """ Hybrid strategy """
        try:
            while not solved:
                print("-----------------------------------------ITERATION:--------------------------------", k+1)
                # milp_model.update()
                # milp_model.tune()
                milp_model.optimize(callbackTerminate)
                print("start time after MILP: {}".format(start_time))
                if k > 0:
                    print("After getting integer cuts %i:" %(k + 1))

                if milp_model.status == grb.GRB.Status.OPTIMAL:
                    """ Check the feasibility subproblem by MILP or CP model """
                    model, next_sequence, next_ts, job_pairs_apos, z_apos, U = self._feasi_by_milp(job_num, machine_num, job_ids, request_times, due_times,
                                                process_intervals, assign, start_time)
                    qubo, offset = model.to_qubo()
                    bqm = dimod.BinaryQuadraticModel.from_qubo(qubo, offset)
                    # sampler = EmbeddingComposite(DWaveSampler({"qpu": True}))
                    # response = sampler.sample(bqm)
                    # next_sequence = response.first.sample
                    next_sequence = pyqubo.solve_qubo(qubo)
                    print("y result: {}".format(next_sequence))
                    print("next ts: {}".format(next_ts))
                    # # do IIS (test conflict contraints)
                    # print('The model is infeasible; computing IIS')
                    # milp2_model.computeIIS()
                    # if milp2_model.IISMinimal:
                    #     print('IIS is minimal\n')
                    # else:
                    #     print('IIS is not minimal\n')
                    # print('\nThe following constraint(s) cannot be satisfied:')
                    # for c in milp2_model.getConstrs():
                    #     if c.IISConstr:
                    #         print('%s' % c.constrName)
                    # Infeasible schedule
                    print(z_apos, job_pairs_apos)
                    next_start_time = milp_model.getAttr("X", start_time)
                    print(next_start_time)

                    check_feasible = True
                    for (i,j) in job_pairs_apos:
                        if z_apos[j] == z_apos[i] and next_sequence['y_(%d,%d)' % (i,j)] == 1:
                            next_start_time[j] = next_start_time[i] + process_intervals[i][z_apos[i]]
                    print("new: {}".format(next_start_time))
                    # Case: same machine, same start time --> not feasible solution
                    for (i,j) in job_pairs_apos:
                        if z_apos[j] == z_apos[i] and next_start_time[i] == next_start_time[j]:
                            check_feasible = False
                    print("check feasible: {}".format(check_feasible))
                    # feasible_schedule = {}
                    # for (i,j) in job_pairs_apos:
                    #     feasible_schedule[(i,j)] = 0
                    #     if z_apos[j] == z_apos[i] and next_sequence['y_(%d,%d)' % (i,j)] == 1 and next_sequence['y_(%d,%d)' % (i,j)] + next_sequence['y_(%d,%d)' % (j,i)] == 1:
                    #         print(i,j, process_intervals[i][z_apos[i]])
                    #         condition = next_start_time[j] < next_start_time[i] + process_intervals[i][z_apos[i]] - U*(1 - next_sequence['y_(%d,%d)' % (i,j)])
                    #         print("here: {}".format(condition))
                    #         if condition:
                    #             continue
                    #         else:
                    #             feasible_schedule[(i,j)] = 1
                    #
                    # print(feasible_schedule)
                    # print(next_start_time)

                    # if any(value==1 for key,value in feasible_schedule.items()):
                    #     check_feasible = 1
                    # print(check_feasible)
                    # print(next_start_time)

                    if not check_feasible:
                        # status2_str = StatusDict[milp2_model.status]
                        # print("Feasibility was stopped with status %s" %status2_str)
                        """ Infeasible """
                        k += 1
                        """ Adding integer cuts """
                        # 1. Creation of set of elements x[i,m] == 1
                        set_xim = [i for i in range(job_num) for m in range(machine_num) if
                                   assign[(i,m)].x == 1]
                        print(set_xim)
                        # 2. Add constraint to MILP model
                        milp_model.addConstr(lhs=grb.quicksum([assign[(i,m)] for i in set_xim
                                                              for m in range(machine_num)
                                                              if assign[(i,m)].x == 1]),
                                             sense=grb.GRB.LESS_EQUAL,
                                             rhs=len(set_xim) - 1,
                                             name="integer cut")
                        # # do IIS (test conflict contraints)
                        # print('The model is infeasible; computing IIS')
                        # milp_model.computeIIS()
                        # if milp_model.IISMinimal:
                        #     print('IIS is minimal\n')
                        # else:
                        #     print('IIS is not minimal\n')
                        # print('\nThe following constraint(s) cannot be satisfied:')
                        # for c in milp_model.getConstrs():
                        #     if c.IISConstr:
                        #         print('%s' % c.constrName)
                    else:
                        """ Feasible """
                        solved = True
                        print("Optimal Schedule Cost: %i" % (milp_model.objVal))
                        milp_model.printStats()
                        self._formulate_schedules(all_machines, job_ids, request_times, due_times,
                                                  process_intervals, assign, next_start_time)
                else:
                    status_str = StatusDict[milp_model.status]
                    print("Optimization was stopped with status %s" %status_str)
                    milp_model.params.DualReductions = 0
                    milp_model._vars = assign, start_time
                    milp_model.Params.lazyConstraints = 1
                    solved = True

        except grb.GurobiError as e:
            print("Error code " + str(e.errno) + ": " + str(e))

        return solved

    """ Feasibility by MILP model """
    def _feasi_by_milp(self, job_num, machine_num, job_ids, r_times, d_times, p_intervals, assign, start_time):
        # model, next_sequence, next_ts = self._create_model(job_num, machine_num, job_ids, r_times, d_times,
        #                                                    p_intervals, assign, start_time)
        model, y, next_ts, job_pairs_apos, z_apos, U = self._create_model(job_num, machine_num, job_ids, r_times, d_times,
                                                                       p_intervals, assign, start_time)

        # """ Write a log file, cannot call in main() function """
        # output_file = os.getcwd() + "/logs/hybrid/2nd/hybrid-jss-milp-milp-" + file_name + ".log"
        # model.setParam(grb.GRB.Param.LogFile, output_file)

        return model, y, next_ts, job_pairs_apos, z_apos, U

    """ Formulate the result """
    def _formulate_schedules(self, all_machines, job_ids, request_times, due_times, process_intervals, assign, start_time):
        """
        variable assign is actually a true feasible assignment, this variable is different from that of
        original solver in gurobi-jss.py
        """
        # print("start-time from MILP: ", start_time)
        # print("sequence from 2nd MILP:", sequence)

        start_times = np.zeros(len(job_ids))
        assign_list = list()
        sequence_list = list()

        for i, j_id in enumerate(job_ids):
            self.schedules[j_id] = dict()
            if not isinstance(start_time[j_id], float):
                self.schedules[j_id]["start"] = start_time[j_id].X
            else:
                self.schedules[j_id]["start"] = start_time[j_id]

            for m in all_machines:
                if assign[j_id, m].x == 1:
                    self.schedules[j_id]["machine"] = m
                    if not isinstance(start_time[j_id], float):
                        self.schedules[j_id]["finish"] = start_time[j_id].x + process_intervals[j_id][m]
                    else:
                        self.schedules[j_id]["finish"] = start_time[j_id] + process_intervals[j_id][m]
                    assign_list.append((j_id, m))
            start_times[i] = start_time[j_id]

        # sequence of jobs vs jobs, sequence of (job, machine) like in original_main.py
        for i_id in job_ids:
             for j_id in job_ids:
                 if i_id < j_id:
                     sequence_list.append((i_id, j_id))

        print("start times of jobs: ", start_times)
        print("assign list of jobs to machines: ", assign_list)
        self.sequence = job_ids[np.argsort(start_times)]

        return

    """ Creation of quantum annealing model with constraints """
    def _create_model(self, job_num, machine_num, job_ids, r_times, d_times, p_intervals, assign,
                      prev_start_time):
        """ Set I' is set of jobs assigned to machines, their assign variable equal to 1"""
        print("assignment:", assign)
        print("previous start time:", prev_start_time)
        # set_I_apos = set S' = {i,j | x_im = x_jm = 1}
        set_I_apos = [i_id for i_id in range(job_num) for m_id in range(machine_num) if
                      assign[(i_id, m_id)].x == 1]
        # print("set I apos:", set_I_apos)
        z_apos = {i_id: m_id for i_id in range(job_num) for m_id in range(machine_num) if
                  assign[(i_id, m_id)].x == 1}
        print(z_apos)

        """ Prepare the index for decision variables """
        # start time of process
        jobs = tuple(job_ids)
        # print("inside:", jobs)
        # sequence of processing jobs: tuple list
        job_pairs_apos = [(i, j) for i in set_I_apos for j in set_I_apos if i != j and z_apos[i] == z_apos[j]]
        # print(job_pairs_apos)
        # # assignment of jobs on machines
        # job_machine_pairs = [(i, m) for i in jobs for m in machines]
        # print(job_machine_pairs)

        """ Parameters model (dictionary) """
        # 1. release time
        release_time = dict(zip(jobs, tuple(r_times)))
        # print("release time:", release_time)
        # 2. due time
        due_time = dict(zip(jobs, tuple(d_times)))
        # print("due time:", due_time)
        # 3. processing time
        process_time = dict(zip(jobs, tuple(p_intervals)))
        print("process time:", process_time)
        U = sum([max(p_intervals[i]) for i in range(job_num)])
        print("test U:", U)

        # for (i,j) in job_pairs_apos:
        #     print("job apos:", i,j)
        #     print(process_time[i][z_apos[i]] - max(due_time.values()))

        """ Create decision variables """
        # # 1. Sequence (Order) of executing jobs
        # y = model.addVars(job_pairs_apos, vtype=grb.GRB.BINARY, name="sequence")
        # # 2. Start time of executing each job (ts = time_start)
        # ts = model.addVars(set_I_apos, lb=0, name="start_time")

        y = {item: pyqubo.Binary("y_(%d,%d)" % item) for item in job_pairs_apos}
        print("y is: {}".format(y))

        """ Create the objective function """
        # model.setObjective(0, sense=grb.GRB.MINIMIZE)
        H = 0
        for (i, j) in job_pairs_apos:
            H += (1 - y[(i, j)] - y[(j, i)] + 2 * y[(i, j)] * y[(j, i)] + y[(i, j)] *
                  (U * (prev_start_time[i].x - prev_start_time[j].x) +
                   sum(process_time[i][m] * assign[(i, m)].x for m in range(machine_num) if assign[(i, m)].x == 1)))
        print("H is {}".format(H))

        # """ Create constraints """
        # # 1. job release time constraint
        # model.addConstrs((ts[i] >= release_time[i] for i in set_I_apos), name="assigned job release constraint")
        # # 2. job due time constraint
        # model.addConstrs((ts[i] <= due_time[i] - process_time[i][z_apos[i]] for i in set_I_apos),
        #                  name="assigned job due constraint")
        # # 3. when assigned, either job 'i' is processed before job 'j' or vice versa
        # model.addConstrs((y[(i,j)] + y[(j,i)] == 1 for (i,j) in job_pairs_apos if i > j and
        #                   assign[(i,z_apos[i])].x == assign[(j,z_apos[j])].x), name="sequence of assigned jobs")
        # # 4. valid cut, starting times, using latest due date as big-M parameter
        # model.addConstrs((ts[j] >= ts[i] + process_time[i][z_apos[i]] - max(due_time.values())*(1 - y[(i,j)])
        #                   for (i,j) in job_pairs_apos if z_apos[j] == z_apos[i]), name="valid cut by big-M")

        # return model, y, ts

        """ Create model """
        model = H.compile()

        ts = prev_start_time

        return model, y, ts, job_pairs_apos, z_apos, U


if __name__ == '__main__':
    StatusDict = {getattr(grb.GRB.Status, s): s for s in dir(grb.GRB.Status) if s.isupper()}

    """ Read data """
    file_name, job_num, machine_num, processing_cost, process_intervals, request_times, due_times = read_data()

    # ID's jobs
    job_ids = np.arange(0, job_num, 1, dtype=np.int32)
    all_jobs = range(job_num)
    all_machines = range(machine_num)

    ssjsp_solver = SSJSP_Hybrid()
    solved = ssjsp_solver.solve(file_name, all_jobs, job_num, all_machines, machine_num, job_ids, request_times,
                                due_times, process_intervals, processing_cost)

    if solved:
        print("Schedules: ", ssjsp_solver.schedules)
        print("Sequence after sorted by increasing start time: ", ssjsp_solver.sequence)

        job_dict = formulate_jobs_dict(job_ids, request_times, process_intervals)
        gantt_chart_plot(job_dict, ssjsp_solver.schedules, processing_cost, "Hybrid Strategy of MILP and MILP")
        plt.show()
