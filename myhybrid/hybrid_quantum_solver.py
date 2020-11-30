#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  13 22:38:43 2020

@author: hoale
"""

""" 
This file contains solver for feasibility problem solved by D-Wave Ocean SDK (2nd quantum-based annealing subproblem)
"""

import numpy as np
import dwavebinarycsp
from pyqubo import Binary


global next_state
global prev_state
global sum_process_time
global U


"""Create label for variables"""
def get_labels(varName, varIndex):
    return f"{varName}_{varIndex}"

"""constraint: sum of variable values in arg anArray must equal 1"""
def c_sumVars_equals_1(*args):
    anArray = np.array(args)
    ssum = 0
    for i in range(anArray.size):
        #print(i, anArray[i])
        ssum = ssum + anArray[i]
    #print("sum: ", ssum)
    if ssum == 1:
        return True
    else:
        return False

def c_jobs_sequence_same_machine(compared_number):
    if compared_number < 1:
        return lambda x: x == 0
    else:
        return lambda x: True

""" Creation of CSP model with constraints """
def _create_model(job_num, machine_num, job_ids, r_times, d_times, p_intervals, assign, prev_start_time):
    """ Set I' is set of jobs assigned to machines, their assign variable equal to 1"""
    # for i_id in range(job_num):
    #     for m_id in range(machine_num):
    #         if assign[(i_id, m_id)].x == 1:
    #             set_I_apos.append(i_id)
    set_I_apos = [i_id for m_id in range(machine_num) for i_id in range(job_num) if assign[(i_id, m_id)].x == 1]
    z_apos = {i_id: m_id for m_id in range(machine_num) for i_id in range(job_num) if assign[(i_id, m_id)].x == 1}
    print(z_apos)
    for i_id in range(job_num):
        print(prev_start_time[i_id].x)
    print("start time from GUROBI: {}".format(prev_start_time))

    """ Prepare the index for decision variables """
    # start time of process
    jobs = tuple(job_ids)
    print("inside:", jobs)
    machines = tuple(range(machine_num))
    print(machines)
    # sequence of processing jobs: tuple list
    job_pairs_apos = [(i, j) for i in set_I_apos for j in set_I_apos if i != j]
    print(job_pairs_apos)
    # assignment of jobs on machines
    job_machine_pairs = [(i, m) for i in jobs for m in machines]
    print(job_machine_pairs)
    # dissimilar parallel machine-machine pair
    machine_pairs = [(m, n) for m in machines for n in machines if m != n]

    """ Parameters model (dictionary) """
    # 1. release time
    release_time = dict(zip(jobs, tuple(r_times)))
    print("release time:", release_time)
    # 2. due time
    due_time = dict(zip(jobs, tuple(d_times)))
    print("due time:", due_time)
    # 3. processing time
    process_time = dict(zip(jobs, tuple(p_intervals)))
    print("process time:", process_time)
    # # 4. processing cost
    # job_cost = dict(zip(jobs, tuple(p_cost)))
    # print("processing cost:", job_cost)
    # 5. define BigU
    for i in range(job_num):
        print(max(p_intervals[i]))
    U = sum([max(p_intervals[i]) for i in range(job_num)])
    print("test U:", U)

    """ Create model """
    # model = grb.Model("SSJSP")
    csp = dwavebinarycsp.ConstraintSatisfactionProblem(dwavebinarycsp.BINARY)

    """ Create decision variables """
    # 1. Assignments of jobs on machines
    # x = model.addVars(job_machine_pairs, vtype=grb.GRB.BINARY, name="assign")
    # 2. Sequence (Order) of executing jobs
    # y = model.addVars(job_pairs_apos, vtype=grb.GRB.BINARY, name="sequence")
    y = {}
    for idx_y in job_pairs_apos:
        y[idx_y] = []
        y[idx_y].append(get_labels("sequence", idx_y))
    print("sequence: {}".format(y))
    # y = {(i,j): Binary(label=f"sequence_{(i,j)}") for (i,j) in job_pairs_apos}
    # print("y variables: {}".format(y))
    # 3. Start time of executing each job (ts = time_start)
    # ts = model.addVars(jobs, lb=0, name="start_time")
    # ts = []
    # for idx_job in jobs:
    #     ts.append(get_labels("start_time", idx_job))
    # print("starting time: {}".format(ts))

    """ Create the objective function """
    # model.setObjective(0)

    """ Create constraints """
    # 1. job release time constraint
    # model.addConstrs((ts[i] >= release_time[i] for i in set_I_apos), name="assigned job release constraint")
    # for i in set_I_apos:
    #     assigned_job_constraints = dwavebinarycsp.Constraint.from_func(
    #         assigned_job_func,
    #         ts[i],
    #         dwavebinarycsp.BINARY,
    #         name=f"assigned_job_release_constraint_{(i)}"
    #     )

    # 2. job due time constraint
    # model.addConstrs((ts[i] <= due_time[i] - process_time[i][z_apos[i]] for i in jobs), name="assigned job due constraint")
    # # 3. one job is assigned to one and only one machine
    # model.addConstrs((grb.quicksum([x[(i,m)] for m in machines]) == 1 for i in jobs),
    #                  name="job non-splitting constraint")
    # # 4. job 'j' is processed after job 'i' when both jobs are assigned to same machine
    # model.addConstrs((y[(i,j)] + y[(j,i)] >= x[(i,m)] + x[(j,m)] - 1 for m in machines for (i,j) in job_pairs if j > i),
    #                   name="assignment-sequencing vars constraint")
    # # 5. sequencing constraint
    # model.addConstrs((ts[j] >= ts[i] + grb.quicksum([process_time[i][m]*x[(i,m)] for m in machines])
    #                   - U*(1 - y[(i,j)]) for (i,j) in job_pairs),
    #                  name="sequence constraint")
    # 6. when assigned, either job 'i' is processed before job 'j' or vice versa
    # model.addConstrs((y[(i,j)] + y[(j,i)] == 1 for (i,j) in job_pairs_apos if i > j), name="sequence of assigned jobs")
    for (i,j) in job_pairs_apos:
        if i > j and z_apos[i] == z_apos[j]:
            sequence_assigned_jobs_constrs = dwavebinarycsp.Constraint.from_func(
                c_sumVars_equals_1,
                y[(i,j)] + y[(j,i)],
                dwavebinarycsp.BINARY,
                name=f"sequence of assigned jobs {(i,j)}"
            )
            csp.add_constraint(sequence_assigned_jobs_constrs)

    # # 7. sequencing varibles = 0 when job 'i' and 'j' are assigned to different machines
    # model.addConstrs((y[(i,j)] + y[(j,i)] + x[(i,m)] + x[(j,n)] <= 2
    #                   for (m,n) in machine_pairs for (i,j) in job_pairs if j > i),
    #                  name="different machine constraint")
    # 8. valid cut, starting times, using latest due date as big-M parameter
    # model.addConstrs((ts[j] >= ts[i] + process_time[i][z_apos[i]] - max(due_time.values())*(1 - y[(i,j)])
    #                   for (i,j) in job_pairs_apos if z_apos[j] == z_apos[i]), name="valid cut by big-M")
    for (i,j) in job_pairs_apos:
        if z_apos[j] == z_apos[i] and i > j:
            print("machine: {}".format(z_apos[i]))
            print("i: {}".format(i))
            for m in range(machine_num):
                print("process time: {}".format(process_time[i][m]))
            print("assign: {}".format(assign))
            # compared_value = prev_start_time[j].x - prev_start_time[i].x - sum(process_time[i][m] for m in range(machine_num) if assign[(i,m)].x == 1)
            # print(compared_value)
            # listVars = [prev_start_time[j].x] + [prev_start_time[i].x] + [sum(process_time[i][m] for m in range(machine_num) if assign[(i,m)].x == 1)] + [U] + y[(i,j)]
            # print(listVars)
            compared_number = (U - prev_start_time[i].x - sum(process_time[i][m] for m in range(machine_num) if
                                                         assign[(i,m)].x == 1) + prev_start_time[j].x) / U
            sequence_jobs_same_machine_constrs = dwavebinarycsp.Constraint.from_func(
                c_jobs_sequence_same_machine(compared_number),
                y[(i,j)],
                dwavebinarycsp.BINARY,
                name=f"same machine, sequence of jobs {(i,j)}"
            )
            csp.add_constraint(sequence_jobs_same_machine_constrs)

    # # QUBO form from CSP formulation
    # dwave_qubo = dwavebinarycsp.stitch(csp)
    # print("DWave QUBO: {}".format(dwave_qubo))

    # set of jobs that are processed
    S_job_set = []
    for m in range(machine_num):
        for (i,j) in job_pairs_apos:
            if assign[(i,m)].x==assign[(j,m)].x==1:
                S_job_set.append((i,j))
    print("S set: {}".format(S_job_set))

    # QUBO form from pyqubo library
    # exp = sum(1-y[(i,j)]-y[(j,i)]+2*y[(i,j)]*y[(j,i)]+y[(i,j)]*(U*(prev_start_time[i].x-prev_start_time[j].x)+sum(process_time[i][m]*assign[(i,m)].x for m in range(machine_num) if assign[(i,m)].x == 1)) for (i,j) in S_job_set)
    # model = exp.compile()
    # print(model)
    # qubo, offset = model.to_qubo()
    # print(qubo, offset)



    print(csp.constraints)

    """CSP to BQM conversion"""
    bqm = dwavebinarycsp.stitch(csp=csp)
    # bqm = model.to_bqm()
    print(bqm)

    # return model, y, ts
    # return model, bqm, y
    return csp, bqm, y
