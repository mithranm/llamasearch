import sys
import importlib
from llamasearch.trustworthiness.linkChecker import (
    extract_domain,
    check_links_domain,
    check_links_end
)

def extractDomainTest1():
    return True

def extractDomainTest2():
    return True

def extractDomainTest3():
    return True

def checkLinksDomainTest1():
    return True

def checkLinksDomainTest2():
    return True

def checkLinksDomainTest3():
    return True

def main():
    passedTests = 0
    failedTests = 0
    
    if (extractDomainTest1 is True):
        passedTests += 1
    else:
        failedTests += 1

    if (extractDomainTest2 is True):
        passedTests += 1
    else:
        failedTests += 1
    
    if (extractDomainTest3 is True):
        passedTests += 1
    else:
        failedTests += 1

    if (checkLinksDomainTest1 is True):
        passedTests += 1
    else:
        failedTests += 1
    
    if (checkLinksDomainTest2 is True):
        passedTests += 1
    else:
        failedTests += 1
    
    if (checkLinksDomainTest3 is True):
        passedTests += 1
    else:
        failedTests += 1

    total = passedTests/ (passedTests+failedTests)
    print(f"Number of Passed Tests: {passedTests}")
    print(f"Number of Failed Tests: {failedTests}")
    print(f"Cases Passed Percentage: {total:.2f}")

if __name__ == "__main__":
    main()

