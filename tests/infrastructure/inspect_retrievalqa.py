import inspect
from langchain.chains import RetrievalQA

def dump_retrievalqa_info():
    print('RetrievalQA init signature:')
    print(inspect.signature(RetrievalQA))
    print('\nInput key default:', RetrievalQA.input_key if hasattr(RetrievalQA, 'input_key') else 'N/A')

if __name__ == "__main__":
    dump_retrievalqa_info() 