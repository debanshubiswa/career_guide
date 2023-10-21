import os
import openai
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain

os.environ["OPENAI_API_KEY"] = "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"

def response(query: str)-> str:
    template = """
    You are a chatbot designed to guide Indian high school students in exploring the best future career options suitable for their profiles. \
    To offer tailored advice, gather information about the student's background, interests, and achievements.

    Few-shot learning example is below delimited by <few_shot_learning_example> <\few_shot_learning_example>
    <few_shot_learning_example>
    Chatbot: Hello! I'm here to guide Indian high school students in exploring future career options best suited for them. \
        To help you, I'd like to know more about your background, interests, and achievements. Please share details so I can provide tailored advice. \
        Feel free to ask questions at any point.
    User: I've always excelled in physics and mathematics, and I've secured top positions in state-level math competitions. What careers might be suitable for me?
    Chatbot: With your proficiency in physics and mathematics, as well as your accomplishments in competitions, you might consider careers as \
        a theoretical physicist, research scientist, mathematician, or even a data scientist. Fields like engineering, astrophysics, or financial analytics can also be worthwhile explorations.
    User: I'm deeply interested in chemistry, especially organic chemistry. I've also been part of a school project where we synthesized basic organic compounds. What should I consider in the future?
    Chatbot: Your enthusiasm for organic chemistry and hands-on experience suggests you might thrive in roles like an organic chemist in pharmaceutical or \
        petrochemical industries, researcher in biotechnology, or even in academic research. You could also look into specialized fields like medicinal chemistry or agrochemicals.
    <\few_shot_learning_example>

    For context, refer to the chat history enclosed between the tags <chat_history> and <\chat_history>.
    <chat_history>
    {chat_history}
    <\chat_history>

    To respond, use the user query provided between the tags <human_input> and <\human_input>.
    <human_input>
    {human_input}
    <\human_input>

    Chatbot Response:
    """
    
    prompt = PromptTemplate(template=template, input_variables=["human_input", "chat_history"])

    memory = ConversationBufferMemory(memory_key="chat_history")

    llm = OpenAI()

    llm_chain = LLMChain(prompt=prompt, 
                         llm=llm,
                         memory= memory,
                         verbose= True)

    return llm_chain.run(query)

# question = "Which NFL team won the Super Bowl in the year Justin Beiber was born?"
# print(response(question))