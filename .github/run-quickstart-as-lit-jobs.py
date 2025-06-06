import json
from datetime import datetime
import glob
import os.path

from lightning_sdk import Studio, Job, Machine, Status


def main():
    print("Creating studio...")
    s = Studio("thunder-quickstarts", "oss-thunder", org="lightning-ai", create_ok=True)
    print("Uploading package and scripts...")
    s.upload_folder("dist", remote_path="dist")
    pkg_path = glob.glob("dist/*.whl")[0]
    s.upload_folder("examples/quickstart", remote_path="quickstart")

    print("Starting studio...")
    s.start()
    print("Installing Thunder and other requirements...")
    s.run(f"pip install {pkg_path} -U -r quickstart/requirements.txt")

    ls_quickstart = glob.glob("examples/quickstart/*.py")
    print("Found quickstart scripts:", ls_quickstart)

    print("Running quickstart scripts...")
    timestamp = datetime.now().strftime("%Y-%m-%d|%H:%M:%S")
    jobs = {
        os.path.basename(script): Job.run(
            name=f"ci-{timestamp}_{script}",
            command=f"pip list && python quickstart/{os.path.basename(script)}",
            studio=s,
            machine=Machine.L40S,
            interruptible=True,
        )
        for script in ls_quickstart
    }

    print("Stopping studio...")
    s.run("rm -rf dist/")
    s.stop()

    print("Waiting for jobs to finish...")
    report, failures = {}, {}
    for name, job in jobs.items():
        job.wait()
        print(f"[{job.status}]\t {job.name}")
        report[name] = str(job.status).lower()
        if job.status != Status.Completed:
            failures[name] = job.logs

    with open("quickstart_report.json", "w") as fp:
        json.dump(report, fp, indent=4)

    print("Showing logs of failed jobs...")
    separator = "=" * 80
    for name, logs in failures.items():
        offset = "=" * (80 - 5 - 2 - len(name))
        print(f"{separator}\n===== {name} {offset}\n{separator}")
        print(logs)
        print(separator + "\n" * 5)

    assert not failures

    # print("Cleaning up...")
    # s.delete()


if __name__ == "__main__":
    main()
