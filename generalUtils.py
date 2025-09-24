from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain
from langchain.prompts import PromptTemplate
from gtts import gTTS
import base64
import re

#Defining max tokens
MAX_TOKENS = 6000

def chunk_documents(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)
    return split_docs

def summarize_chain(docs, llm):
    total_text = " ".join(doc.page_content for doc in docs)
    total_tokens = len(total_text) // 4       # Taking rough estimate: 1 token ≈ 4 characters for English

    # Use 'stuff' for summarization if under token limit else switch to 'map-reduce'
    if total_tokens < MAX_TOKENS:
        # Prompt setup
        template = ('''Please provide a concise and detailed summary of the following content.
                    Understand the type and message of the text provided.
                    Add suitable title followed by introduction.  
                    Keep section-wise brief pointers (mentioning topics or highlights).
                    End with a fitting conclusion.
                    Text: {text}''')

        prompt = PromptTemplate(input_variables=['text'], template=template)
        chain = load_summarize_chain(llm, chain_type="stuff", prompt=prompt)
        return chain.run(docs)
    else:
        chunked_docs = chunk_documents(docs)

        initial_template = ("You are an assistant for text summarization tasks. "
                    "Write a concise and short summary of the provided text. \n"
                    "{text}"
                )

        initial_prompt = PromptTemplate(input_variables=['text'], template = initial_template)

        final_template = '''Provide the final summary of the entire text with these important points.
                        Add a suitable title. Start the precise summary with an introduction, state key notes in pointers and 
                        end with conclusion.
                        The provided text: {text}
                    '''
        final_prompt = PromptTemplate(input_variables=['text'], template=final_template) 
        
        chain = load_summarize_chain(llm, chain_type="map_reduce", map_prompt=initial_prompt, combine_prompt=final_prompt)
        return chain.run(chunked_docs)
    
def generate_audio(summary_text, lang="en"):

    #Formatting text for better audio
    text = re.sub(r'[#*_>`\-]', '', summary_text)       # Simplifying formatting of markdown text
    text = re.sub(r'(?<=[^\.\!\?])\n', '. ', text)      # Add periods if line ends without one
    text = re.sub(r'\n+', ' ', text)                    # Flatten newlines
    text = re.sub(r'\s{2,}', ' ', text)                 # Remove extra spaces

    
    tts = gTTS(text, lang=lang)
    tts.save("summary_audio.mp3")
    with open("summary_audio.mp3", "rb") as f:
        audio_bytes = f.read()

    b64 = base64.b64encode(audio_bytes).decode()
    return audio_bytes, b64
    



    