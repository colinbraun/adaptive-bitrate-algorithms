# Lab Project 3: ABR
This directory contains the lab project 3 files.

## Project Goals and Background
This project pertains to Adaptive Bit Rate (ABR) algorithms for Internet video delivery. The project will involve reading papers published in top scientific conferences on computer networking, and implementing and evaluating ABR algorithms proposed in them on a custom ABR simulator that we have built. You will also critique these algorithms, identify when they work well and when they do not, explore design variants, and turn in your results and findings in a research report that you will submit.

## Simulator Code Structure
The code structure is as follows,

1. simulator.py: This is the top-level simulator file. It parses the test case and simulates the viewing session using the choices from your ABR algorithm. DO NOT MODIFY.
2. tester.py: This script will run simulator.py for all test cases in the tests/ directory. DO NOT MODIFY.
3. Classes/: This contains supporting code for the simulator. DO NOT MODIFY.
4. student/student1.py: This is where you are to implement your first ABR algorithm. It contains a predefined class “ClientMessage” containing all the various metrics that might be used by your algorithm. Fill out the student_entrypoint() section of this file.
5. student/student2.py: This is where you are to implement your second ABR algorithm.
6. student/studentX.py: If you would like to implement more algorithms, you may copy over student1.py or student2.py to make a student3, student4, .... and run them the same way.

## Helper Functions and Global Variables
Because the student code is called from one function (student_entrypoint()), you are encouraged to implement any necessary classes, helper functions, and global variables in the studentX.py classes.
Some algorithms, such as RobustMPC, require knowledge of previous chunks in order to make predictions on future chunks. To capture this behavior, you should save any necessary information in global or module-level variables between student_entrypoint() calls. If you do not know how to do this, search online for a tutorial on the Python “global” keyword.

## Submitting your Code
When submitting your code, you are to submit one algorithm in “student1.py” and another algorithm in “student2.py”. For example, if implementing MPCSigcomm15 algorithm, and the BBA-2 algorithm then MPCSigcomm15 algorithm is in “student1.py” and BBA-2 algorithm is in “student2.py”.

## Running your Code
The simulator runs your algorithm and outputs statistics on a per-chunk basis as well as for the complete video as a whole. You can run the simulator with the following command:

```bash 
python simulator.py <path to the test file (.ini)> <Student algorithm to run> -v
```

The path to the test file should be the path to one of the .ini files in the tests/ directory. The student algorithm to run should be an integer 1 or 2 (or higher if you made more algorithms) to run student1.py or student2.py. ‘-v’ is an optional flag that enables verbose output for the simulation. This prints the download times and quality selections for each chunk.

For example, running:
```bash
python simulator.py tests/hi_avg_hi_var.ini 2 -v
```

will start the simulator running the test "hi_avg_hi_var.ini" using the algorithm in student2.py and enable verbose logging.

The tester will run your algorithm and output statistics for all test cases. It is called with
```bash
python tester.py <Student algorithm to run (1 or 2)>
```

## To submit:
----------
Create a tag named "submission" when you are ready to submit. You can use the following command or the github interface to do so.

``` bash
git tag -a submission -m "optional message"
```

This will show you if the tag is created:

```bash
git tag
```

Do not submit any binaries. Your git repo should only contain source files; no products of compilation.

Once you submit, a file "grade.txt" will appear in the branch 'grade' after a few minutes.
This file does not reflect your final grade. This grade is 0 if one of the test cases ended in an compile/run-time error in the simulator and 1 otherwise. Your actual grade will be heavily based on the criteria below and the report you submit.

## Grading

The project has open-ended components. Getting a decent grade will require implementing (i) both the RobustMPC and BBA-2 algorithms, and a variant of each; and (ii) reporting results  clearly and in a well thought out manner, presenting good quality graphs, and clearly interpreting results, However, the very best grades will be obtained by students that explore particularly new and interesting variants of these algorithms, and show creativity, effort and initiative in the design and implementation of the variants, and in the open-ended components. We may award a bonus to students that go particularly beyond the norm in terms of the open-ended components, and exhibit a high degree of passion and effort in the project. Note that the bar for a bonus will be high and subjective.
