import inspect
import textwrap
from langchain.chains.retrieval_qa.base import RetrievalQA

print(textwrap.dedent(inspect.getsource(RetrievalQA._call))) 