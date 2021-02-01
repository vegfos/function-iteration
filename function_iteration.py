import numpy as np
import streamlit as st

def initialization(n, w_low, w_high, c_low, c_high, m):
    # Initial guess for value function
    v = np.zeros(n)

    # Set up state space approximiation
    w = np.linspace(w_low, w_high, n)

    # Discrete controls
    c_possible = np.linspace(c_low, c_high, m)

    return v, w, c_possible

def capital_transition(r, c_feasible, wealth):
    c_feasible = np.asarray(c_feasible)
    return np.asarray(r * (wealth - c_feasible))

def bellman(c_feasible, u, b, v_next):
    c_feasible = np.asarray(c_feasible)
    v_next = np.asarray(v_next)
    return u(c_feasible)+(b*v_next)

def closest_neighbor(a, b):
    a = np.asarray(a)
    b = np.asarray(b)

    dists = np.abs(np.subtract.outer(a, b))
    pos = np.argmin(dists, axis=1)

    return pos

def function_iteration(b, r, u, w_low, w_high, n, c_low, c_high, m, maxit, tol):
    v, w, c_possible = initialization(n, w_low, w_high, c_low, c_high, m)
    v_new, policy = np.zeros(n), np.zeros(n)
    iteration_bar = st.progress(0)

    for i in range(maxit):
        # Iterate over nodes (wealth states)
        iteration_bar.progress(round((i/maxit)*100))
        for j in range(n):
            # Current wealth state
            current_wealth = w[j]

            # Calculate feasible consumption constrainted by non-negativity:
            # Agents cannot eat more than is available in state j
            c_feasible = np.minimum(c_possible, np.full_like(c_possible, current_wealth))            

            # Calculate next period wealth for each possible control value
            w_next = capital_transition(r, c_feasible, current_wealth)

            # Because we are evaluating a discrete space we need
            # for each next period wealth find a corresponding
            # state that is saved in the state space
            # Problem can be generalized to:
            # For arrays A, B. Find array C with elements from B
            # with the minimum deviation between A_i, B_j j in [0, N]
            
            # Create a 2d array dists suchs that 
            # dists[i, j] = A[i] - B[j]
            # Call argmin to find indices with the smallest deviation
            # Return A[indices]
            w_pos = closest_neighbor(w_next, w)

            # Evaluate value function at next period's capital state
            v_next = v[w_pos]

            # Evaluate r.h.s Bellman for all feasible controls
            v_vect = bellman(c_feasible, u, b, v_next)

            # Pick the highest value and corresponding policy
            h_val_index = np.argmax(v_vect)
            v_new[j] = v_vect[h_val_index]
            policy[j] = c_possible[h_val_index]

        if (v_new == v).all():
            st.write('Converged Completely at iteration' + str(i) +"/"+str(maxit))
            return v, w, policy
        elif np.nanmax(abs(v_new - v)) <= tol:
            st.write('Converged within tolerance at iteration ' + str(i)+"/"+str(maxit))

            return v, w, policy
        
        v = np.copy(v_new)

    st.write('Did not converge after maximum iterations (' + str(i) + ')')
    
    return v, w, policy