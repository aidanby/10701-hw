
import argparse
import importlib.util
import glob
import json
import os
import signal
import time



class TimeoutFunctionException(Exception):
    """Exception to raise on a timeout"""
    pass


class TimeoutFunction:
    def __init__(self, function, timeout):
        self.timeout = timeout
        self.function = function

    def handle_timeout(self, signum, frame):
        raise TimeoutFunctionException()

    def __call__(self, *args, **keyArgs):
        # If we have SIGALRM signal, use it to cause an exception if and
        # when this function runs too long.  Otherwise check the time taken
        # after the method has returned, and throw an exception then.
        if hasattr(signal, 'SIGALRM'):
            old = signal.signal(signal.SIGALRM, self.handle_timeout)
            signal.alarm(self.timeout)
            try:
                result = self.function(*args, **keyArgs)
            finally:
                signal.signal(signal.SIGALRM, old)
            signal.alarm(0)
        else:
            startTime = time.time()
            result = self.function(*args, **keyArgs)
            timeElapsed = time.time() - startTime
            if timeElapsed >= self.timeout:
                self.handle_timeout(None, None)
        return result

def run_test(test_name, score_multiplier=1):
    spec = importlib.util.spec_from_file_location("module.name", test_name)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    max_score = mod.max_score() * score_multiplier
    timeout = mod.timeout()

    try:
        if timeout is not None:
            timed_function = TimeoutFunction(mod.test, timeout)
            try:
                score, message = timed_function()
                score *= score_multiplier
            except TimeoutFunctionException:
                score = 0
                message = "Timed out after {} seconds".format(timeout)
        else:
            score, message = mod.test()

    except AssertionError as e:
        score = 0
        if hasattr(e, 'message'):
            message = e.message
        else:
            message = str(e)

    print('Test case -- {}: {}/{} points. {}'.format(test_name, score, max_score, message))

    return max_score, score, message


def grade_question(question_name, report, score_multiplier=1):
    question_max_score = 0
    question_score = 0

    question_output = '{}\n----------\n\n'.format(question_name)

    print(question_name)
    print('----------')

    for test_name in sorted(glob.glob('test_cases/{}/*.py'.format(question_name))):
        question_output += '{}: {} ...\n'.format(question_name, test_name)
        #print('{}: {} ...'.format(question_name, test_name))

        test_max_score, test_score, test_output = run_test(test_name, score_multiplier=score_multiplier)

        question_score += test_score
        question_max_score += test_max_score
        question_output += test_output
        question_output += '{} points: {}/{}\n\n'.format(test_name, test_score, test_max_score)
        
    question_report = {}
    question_report['name'] = question_name
    question_report['max_score'] = question_max_score
    question_report['score'] = question_score
    question_report['output'] = question_output

    report['tests'].append(question_report)

    question_output += 'Points: {}/{}\n\n'.format(question_score, question_max_score)
    print('{} points: {}/{}'.format(question_name, question_score, question_max_score))
    print()
    print('==========')

    return question_max_score, question_score

def grade_all(report, score_multiplier=1):
    total_max_score = 0
    total_score = 0

    for question_name in sorted(glob.glob('test_cases/Q*')):
        _, question_name = os.path.split(question_name)

        question_max_score, question_score = grade_question(
            question_name, report, score_multiplier=score_multiplier)
        
        total_max_score += question_max_score
        total_score += question_score

    print('Autograder complete')
    print('Points {}/{}'.format(total_score, total_max_score))
    print()



def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--test', '-t',
        dest='test',
        help='Run one particular test. Name of *.py file, including relative path.')

    parser.add_argument('--question', '-q',
        dest='question',
        help='Grade one particular question.')

    args = vars(parser.parse_args())

    return args


def save_gradescope_report(report):
    report_filename = 'report.json'
    with open(report_filename, 'w') as report_file:
        report_file.write(json.dumps(report, indent=4))



if __name__ == "__main__":

    args = parse_args()

    report = {}
    report['tests'] = []

    score_multiplier = 1

    if args['test'] is not None:
        run_test(args['test'], score_multiplier=score_multiplier)

    elif args['question'] is not None:
        grade_question(args['question'], report, score_multiplier=score_multiplier)

    else:
        grade_all(report, score_multiplier=score_multiplier)

    report_filename = 'report.json'
    with open(report_filename, 'w') as report_file:
        report_file.write(json.dumps(report, indent=4))

