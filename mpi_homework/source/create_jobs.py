import argparse

TF = (False, True)

def job_name(nodes, c, i, m):
    name = f"matrixmul_{nodes}_{c}"
    if i:
        name += "_i"
    if m:
        name += "_m"
    return name


def get_command(f, c, i, m, exp):
    cmd = f"srun ../build/matrixmul -f {f} -s 42 -c {c} -e {exp} -g 0"
    if i:
        cmd += " -i"
    if m:
        cmd += " -m"
    return cmd


def create_job(path, f, nodes, c, i, m):
    name = job_name(nodes, c, i, m)

    exp = nodes * 5
    job = open(f"{path}/{name}", "w")

    cmd = get_command(f, c, i, m, exp)
    time = 10

    job.write("#!/bin/bash -l\n")
    job.write(f"#SBATCH --job-name {name}\n")
    job.write(f"#SBATCH --output out/{name}.out\n")
    job.write(f"#SBATCH --error out/{name}.err\n")
    job.write(f"#SBATCH --account \"GC72-18\"\n")
    job.write(f"#SBATCH --nodes {nodes}\n")
    job.write(f"#SBATCH --tasks-per-node 24\n")
    job.write(f"#SBATCH --time 0:{time}:00\n")
    job.write(cmd)


if __name__ == "__main__":
    f = "../../examples/sample_20000_1000"

    m = True

    for nodes in (1, 2, 4, 8, 16, 32):
        for c in (1, 2, 4, 8, 16):
            for i in TF:
                if ((nodes * 24) % (c * c) == 0):
                    create_job("jobs", f, nodes, c, i, m)
