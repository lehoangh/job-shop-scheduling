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
# from myhybrid.hybrid_relaxed_milp import _create_milp_model
from myhybrid.hybrid_quantum_solver import _create_model, get_labels
from gantt_plot import gantt_chart_plot, formulate_jobs_dict
import matplotlib.pyplot as plt
from dwave.system import DWaveSampler, AutoEmbeddingComposite


# """ Creation of relaxed MILP model with constraints (subproblem 1) """
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
    # 4. A valid cut, it tightens LP relaxation, total processing time of all jobs assigned to same machine
    # less than diff of latest due date & earliest release date
    # model.addConstrs((grb.quicksum([x[(i,m)]*process_time[i][m] for i in jobs]) <= max(due_time.values()) -
    #                   min(release_time.values()) for m in machines), name="total processing time of all jobs")

    return model, x, ts


"""Check the feasible schedule condition"""
def check_feasibility(job_num, machine_num, request_times, due_times, prev_start_time, process_time, U, assign, sequence):
    """Taking sequence of jobs in the same machine from the results of QA"""
    job_pairs = [(i,j) for i in range(job_num) for j in range(job_num) if i != j]
    print("pairs of jobs: {}".format(job_pairs))
    new_ts = {k: v.x for k, v in prev_start_time.items()}
    print("new ts: {}".format(new_ts))
    true_sequence = {k: False for k in sequence.keys()}
    sequence_jobs = {}
    for m in range(machine_num):
        sequence_jobs[m] = {}
        for i in range(job_num):
            sequence_jobs[m][i] = []
    print(sequence_jobs)
    print("sequence: {}".format(sequence))

    # create an adjacent jobs connecting to each job
    for m in range(machine_num):
        for (i,j) in job_pairs:
            if assign[(i,m)].x == assign[(j,m)].x == 1 and sequence.get(get_labels("sequence", (i,j))) == 1:
                sequence_jobs[m][i].append(j)
    print("sequence jobs {}".format(sequence_jobs))

    # sorting the sequence of jobs according to the decreasing order of the number of adjacent elements for each job
    sorted_sequence_jobs = {}
    for m in sequence_jobs.keys():
        sorted_sequence_jobs[m] = dict(sorted(sequence_jobs[m].items(), key=lambda x: len(x[1]), reverse=True))
    print("sorted: {}".format(sorted_sequence_jobs))
    jobs_sequence = {}
    for m in sorted_sequence_jobs.keys():
        jobs_sequence[m] = []
        for k in sorted_sequence_jobs[m].keys():
            len_k = len(sorted_sequence_jobs[m][k])
            if len_k > 1:
                jobs_sequence[m].append(k)
            elif len_k == 1:
                jobs_sequence[m].append(k)
                jobs_sequence[m].append(sorted_sequence_jobs[m][k][0])
    print("job sequence same machine: {}".format(jobs_sequence))

    for m in range(machine_num):
        for job_id_prev in range(len(jobs_sequence[m])-1):
            if jobs_sequence[m]:
                print(jobs_sequence[m][job_id_prev])
                if assign[(jobs_sequence[m][job_id_prev+1],m)].x == assign[(jobs_sequence[m][job_id_prev],m)].x == 1:
                    new_ts[jobs_sequence[m][job_id_prev+1]] = new_ts[jobs_sequence[m][job_id_prev]] + process_time[jobs_sequence[m][job_id_prev]][m]
    print("NEW: {}".format(new_ts))

    for m in range(machine_num):
        for (i,j) in job_pairs:
            if assign[(i,m)].x == assign[(j,m)].x == 1:
                print("i,j: {},{}".format(i,j))
                sequence_label_ij = get_labels("sequence", (i,j))
                sequence_label_ji = get_labels("sequence", (j,i))
                print("Label of sequence: {}".format(sequence_label_ij))
                # y_ij + y_ji == 1
                # if sequence.get(sequence_label_ij) + sequence.get(sequence_label_ji) != 1:
                #     print("EXIT HERE")
                #     return False, new_ts
                # else:
                if sequence.get(sequence_label_ij) == 1:
                    if new_ts[j] >= request_times[j] and new_ts[j] <= due_times[j] - process_time[j][m]*assign[(j,m)].x:
                        true_sequence[sequence_label_ij] = True

    if all(true_sequence[k] is True for k in true_sequence.keys() if sequence[k] == 1):
        return True, new_ts
    else:
        return False, new_ts

    #                 print(prev_start_time[i].x, process_time[i][m])
    #                 new_start_time_j = new_ts[i] + process_time[i][m]*assign[(i,m)].x
    #                 print("new starting time for j: {}".format(new_start_time_j))
    #                 if new_start_time_j <= due_times[j] - process_time[j][m]*assign[(j,m)].x and new_start_time_j >= request_times[j]:
    #                 # if prev_start_time[j].x >= prev_start_time[i].x + process_time[i][m]*assign[(i,m)].x - U*(1 - sequence.get(sequence_label_ij)):
    #                     new_ts[j] = new_start_time_j
    #                     true_sequence[sequence_label_ij] = True
    # if all(true_sequence[k] == True for k in true_sequence.keys() if sequence[k] == 1):
    #     return True, new_ts
    # else:
    #     return False, new_ts



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
        print("all of jobs: {}".format(all_jobs))
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
                milp_model.optimize()
                if k > 0:
                    print("After getting integer cuts %i:" %(k + 1))

                if milp_model.status == grb.GRB.Status.OPTIMAL:
                    """ Check the feasibility subproblem by MILP or CP model """
                    U = sum([max(process_intervals[i]) for i in range(job_num)])
                    print("outside test U:", U)
                    # milp2_model, next_sequence, next_ts = self._feasi_by_milp(job_num, machine_num, job_ids,
                    #                                                           request_times, due_times,
                    #                                                           process_intervals, assign,
                    #                                                           start_time)
                    qpu_model, bqm, next_sequence = self._feasi_by_milp(job_num, machine_num, job_ids,
                                                             request_times, due_times,
                                                             process_intervals, assign,
                                                             start_time)

                    print(milp_model.status)
                    # print(milp2_model.status)
                    print(next_sequence)
                    print("Assign: {}".format(assign))
                    print("Start time: {}".format(start_time))
                    # Solve MILP2 model
                    # model.tune()
                    # milp2_model.optimize()
                    """Solve the feasibility problem by a quantum annealer"""
                    qpu_solver = DWaveSampler(solver={"qpu": True})
                    # self.sampler = DWaveSampler(solver={"topology__type": "pegasus", "qpu": True})
                    print("Connected to solver: {}".format(qpu_solver.solver))
                    qpu_sampler = AutoEmbeddingComposite(qpu_solver)
                    samples = qpu_sampler.sample(bqm=bqm)
                    print("y: {}".format(samples))
                    print("sample: {}".format(samples.record["sample"]))
                    for sa in samples.samples():
                        print(sa, type(sa))
                        sa_dict = {var: sa[var] for var in samples.variables}
                        print(sa_dict, type(sa_dict))
                        check_solution = qpu_model.check(sa_dict)
                        print("CSP: {}".format(qpu_model.constraints))
                        print("Check solution: {}".format(check_solution))
                        # decoded_sample = qpu_model.decode_sample(sa_dict, vartype="BINARY")
                        # broken_constraints = decoded_sample.constraints(only_broken=True)
                        # print(broken_constraints)
                        # output_sample = decoded_sample.sample
                        output_sample = sa_dict
                        check, next_ts = check_feasibility(job_num, machine_num, request_times,
                                                           due_times, start_time, process_intervals,
                                                           U, assign, output_sample)
                        print("Check: {}".format(check))
                    # milp2_model.write(os.getcwd() + "/logs/hybrid/2nd/2nd_milp_" + str(k+1) + ".lp")
                    # milp2_model.setParam(grb.GRB.Param.LogFile, os.getcwd() + "/logs/hybrid/2nd/2nd_milp_" +
                    #                      str(k+1) + ".log")
                    # print(milp2_model.status)
                    # print("y : {}".format(next_sequence))
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

                        # if milp2_model.status == grb.GRB.Status.INFEASIBLE:
                        # INFEASIBLE
                        if not check:
                            # status2_str = StatusDict[milp2_model.status]
                            # print("Feasibility was stopped with status %s" %status2_str)
                            print("Feasibility was stopped with status %s" %check)

                            """ Infeasible """
                            k += 1
                            """ Adding integer cuts """
                            # 1. Creation of set of elements x[i,m] == 1
                            set_xim = [i for i in range(job_num) for m in range(machine_num) if
                                       assign[(i,m)].x == 1]
                            # 2. Add constraint to MILP model
                            milp_model.addConstrs((grb.quicksum([assign[(i,m)] for i in set_xim
                                                                  for m in range(machine_num)
                                                                  if assign[(i,m)].x == 1]) <= len(set_xim) - 1
                                                    for m in range(machine_num)), name="integer cut")
                        else:
                            """ Feasible """
                            solved = True
                            print("Optimal Schedule Cost: %i" % milp_model.objVal)
                            milp_model.printStats()
                            self._formulate_schedules(all_machines, job_ids, request_times, due_times,
                                                      process_intervals, assign, next_sequence, next_ts)
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
        # model, next_sequence, next_ts = _create_model(job_num, machine_num, job_ids, r_times, d_times,
        #                                                    p_intervals, assign, start_time)
        model, bqm, next_sequence = _create_model(job_num, machine_num, job_ids, r_times, d_times,
                                                      p_intervals, assign, start_time)

        """ Write a log file, cannot call in main() function """
        output_file = os.getcwd() + "/logs/hybrid/2nd/hybrid-jss-milp-milp-" + file_name + ".log"
        # model.setParam(grb.GRB.Param.LogFile, output_file)

        # return model, next_sequence, next_ts
        return model, bqm, next_sequence

    """ Formulate the result """
    def _formulate_schedules(self, all_machines, job_ids, request_times, due_times, process_intervals, assign,
                             sequence, start_time):
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
            self.schedules[j_id]["start"] = start_time[j_id]

            for m in all_machines:
                if assign[j_id, m].x == 1:
                    self.schedules[j_id]["machine"] = m
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

    # """ Creation of MILP model with constraints """
    # def _create_model(self, job_num, machine_num, job_ids, r_times, d_times, p_intervals, assign,
    #                   prev_start_time):
    #     """ Set I' is set of jobs assigned to machines, their assign variable equal to 1"""
    #     # print("assignment:", assign)
    #     # print("previous start time:", prev_start_time)
    #     set_I_apos = [i_id for i_id in range(job_num) for m_id in range(machine_num) if
    #                   assign[(i_id, m_id)].x == 1]
    #     # print("set I apos:", set_I_apos)
    #     z_apos = {i_id: m_id for i_id in range(job_num) for m_id in range(machine_num) if
    #               assign[(i_id, m_id)].x == 1}
    #     print(z_apos)
    #
    #     """ Prepare the index for decision variables """
    #     # start time of process
    #     jobs = tuple(job_ids)
    #     # print("inside:", jobs)
    #     # sequence of processing jobs: tuple list
    #     job_pairs_apos = [(i, j) for i in set_I_apos for j in set_I_apos if i != j and z_apos[i] == z_apos[j]]
    #     # print(job_pairs_apos)
    #     # # assignment of jobs on machines
    #     # job_machine_pairs = [(i, m) for i in jobs for m in machines]
    #     # print(job_machine_pairs)
    #
    #     """ Parameters model (dictionary) """
    #     # 1. release time
    #     release_time = dict(zip(jobs, tuple(r_times)))
    #     # print("release time:", release_time)
    #     # 2. due time
    #     due_time = dict(zip(jobs, tuple(d_times)))
    #     # print("due time:", due_time)
    #     # 3. processing time
    #     process_time = dict(zip(jobs, tuple(p_intervals)))
    #     # print("process time:", process_time)
    #
    #     # for (i,j) in job_pairs_apos:
    #     #     print("job apos:", i,j)
    #     #     print(process_time[i][z_apos[i]] - max(due_time.values()))
    #
    #
    #     """ Create model """
    #     model = grb.Model("SSJSP")
    #
    #     """ Create decision variables """
    #     # 1. Sequence (Order) of executing jobs
    #     y = model.addVars(job_pairs_apos, vtype=grb.GRB.BINARY, name="sequence")
    #     # 2. Start time of executing each job (ts = time_start)
    #     ts = model.addVars(set_I_apos, lb=0, name="start_time")
    #
    #     """ Create the objective function """
    #     model.setObjective(0, sense=grb.GRB.MINIMIZE)
    #
    #     """ Create constraints """
    #     # 1. job release time constraint
    #     model.addConstrs((ts[i] >= release_time[i] for i in set_I_apos), name="assigned job release constraint")
    #     # 2. job due time constraint
    #     model.addConstrs((ts[i] <= due_time[i] - process_time[i][z_apos[i]] for i in set_I_apos),
    #                      name="assigned job due constraint")
    #     # 3. when assigned, either job 'i' is processed before job 'j' or vice versa
    #     model.addConstrs((y[(i,j)] + y[(j,i)] == 1 for (i,j) in job_pairs_apos if i > j and
    #                       assign[(i,z_apos[i])].x == assign[(j,z_apos[j])].x), name="sequence of assigned jobs")
    #     # 4. valid cut, starting times, using latest due date as big-M parameter
    #     model.addConstrs((ts[j] >= ts[i] + process_time[i][z_apos[i]] - max(due_time.values())*(1 - y[(i,j)])
    #                       for (i,j) in job_pairs_apos if z_apos[j] == z_apos[i]), name="valid cut by big-M")
    #
    #     return model, y, ts


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
