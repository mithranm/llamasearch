import sys
from llamasearch.core.llm_cpu import main

sys.argv = ['llm.py', '--query', "Who is Georgi Zahariev?", '--persist']

main()