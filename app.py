import os
import datetime
import hashlib
from flask import Flask, session, url_for, redirect, render_template, request, abort, flash
import numpy as np

app = Flask(__name__)
app.config.from_object('config')

# global variables
edges = []
nodes = []

def multiply_matrix_vector(matrix, vector):
    result = np.zeros(len(vector), dtype=np.double)
    for idx, row in enumerate(matrix):
        for idx2, cell in enumerate(row):
            result[idx] += cell * vector[idx]
    return result


# functions def
def pagerank(M, num_iterations: int = 100, d: float = 0.85):
    N = M.shape[1]
    v = np.ones(N) / N
    M_hat = (d * M + (1 - d) / N)
    for i in range(num_iterations):
        v = multiply_matrix_vector(M_hat, v)
    return v

def test_col_sums_to_one(A):
    sum = np.vectorize(round)(np.sum(A, axis=0))
    assert list(sum).count(1) + list(sum).count(0) == len(sum)


# program execution
file = open('./static/filtered_data.txt', 'r')
count = 0

while True:
    count += 1

    # Get next line from file
    line = file.readline()
    # if line is empty
    # end of file is reached
    if not line:
        break
    # otherwise get nodes and edges
    edge = line.replace('\n', '').split(",")
    if edge[0] not in nodes: nodes.append(edge[0])
    if edge[1] not in nodes: nodes.append(edge[1])
    edges.append(edge)

file.close()

# create the adjacency matrix
n_nodes = len(nodes)
A = np.zeros((n_nodes,n_nodes), dtype=np.double)

for edge in edges:
    i = nodes.index(edge[0])
    j = nodes.index(edge[1])
    A[j][i] = 1

# set the weight depending on how many outgoing edges the node has
freq = list(np.sum(A, axis=0))
assert len(freq) == len(A)
for idx, f in enumerate(freq):
    if f != 0 and f != 1:
        for i, row in enumerate(A):
            if row[idx] == 1: 
                A[i][idx] = 1 / f

# test if each column sums up to 1
test_col_sums_to_one(A)

# save the adjacency matrix in a file in the disc
f = open("adjacency_matrix.txt", "w")
f.write(str(A.tolist()).replace("[","[\n").replace("]","]\n"))
f.close()

# execute the page rank algorithm
v = pagerank(A, 2, 0.85)
for idx, el in enumerate(v):
    v[idx] = round(el,10)
print(v)

# rendering
@app.route("/")
def FUN_root():
    return render_template("index.html", nodes=enumerate(nodes), nodes2=nodes, edges=enumerate(edges), pageranks=enumerate(v))

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0")
