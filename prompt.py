from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnableLambda

sentimenttemplate = ChatPromptTemplate.from_template(
    """In a single word, either 'positive' or 'negative', 
    provide the overall sentiment of the following piece of text: {text}"""
)
maintopictemplate = ChatPromptTemplate.from_template(
    """Identify and state, as concisely as possible, the main topic 
    of the following piece of text. Only provide the main topic and no other helpful comments. 
    Text: {text}"""
)
followuptemplate = ChatPromptTemplate.from_template(
    """What is an appropriate and interesting followup question that would help 
    me learn more about the provided text? Only supply the question. 
    Text: {text}"""
)
parser = StrOutputParser()
outputformatter = RunnableLambda(lambda responses: (
    f"Statement: {responses['statement']}\n"
    f"Overall sentiment: {responses['sentiment']}\n"
    f"Main topic: {responses['maintopic']}\n"
    f"Followup question: {responses['followup']}\n"
))

prep_for_templates = RunnableLambda(lambda statement: {"text": statement})

sentiment_chain = sentimenttemplate | llm | parser
maintopic_chain = maintopictemplate | llm | parser
followup_chain = followuptemplate | llm | parser
parallel_chain = RunnableParallel({
    "statement": RunnableLambda(lambda x: x["text"]), 
    "sentiment": sentiment_chain,
    "maintopic": maintopic_chain, 
    "followup": followup_chain
})
full_chain = prep_for_templates | parallel_chain | outputformatter
statements = [
    "I had a fantastic time hiking up the mountain yesterday.",
    "The new restaurant downtown serves delicious vegetarian dishes.",
    "I am feeling quite stressed about the upcoming project deadline.",
    "Watching the sunset at the beach was a calming experience.",
    "I recently started reading a fascinating book about space exploration."
]
print("=== LCEL Parallel Chain Results ===\n")
for statement in statements:
    result = full_chain.invoke(statement)
    print(result)
    print("-" * 80)
