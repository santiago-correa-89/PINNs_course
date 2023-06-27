import openai
import os

import panel as pn
pn.extension()

from IPython.display import display, Markdown, Latex, HTML, JSON
from readline import Redlines

from dotenv import load_dotenv, find_dotenv

_ = load_dotenv(find_dotenv())

openai.api_key = os.getenv('openai_api_key')

prod_review = """text"""

prompt = """indications"""

response = get_completion(prompt)

display(Latex(response))
diff = Redlines(prod_review, response)
display(Markdown(diff.output_markdown))
print(response)

inp = pn.widgets.TextInput(value="Hi", placeholder='Enter text here…')
button_conversation = pn.widgets.Button(name="Chat!")
interactive_conversation = pn.bind(collect_messages, button_conversation)
dashboard = pn.Column(inp, pn.Row(button_conversation), pn.panel(interactive_conversation, loading_indicator=True, height=300),
)
dashboard

def get_completion(prompt, model='gpt-3.5-turbo'):
    messages = [{"role": "user", "content": prompt}]
    response = openai.Completion.create(
    model = model,
    messagees = messages,
    temperature = 0, 
    )
    return response.choices[0].message['content']

def get_completion_from_messages(messages, model='gpt-3.5-turbo', temperature = 0):
    response = openai.Completion.create(
    model = model,
    messages = messages,
    temperature = temperature, 
    )
    #print(response.chioce[0].messages)
    return response.choices[0].messages['content']

def collect_messages(_):
    prompt = inp.value_input
    inp.value = ''
    context.append({'role':'user', 'content':f"{prompt}"})
    response = get_completion_from_messages(context)
    context.append({'role':'assistant', 'content':f"{response}"})
    panels.append(pn.Row('User:', pn.pane.Markdown(prompt, width=600)))
    panels.append(pn.Row('Assistant:', pn.pane.Markdown(response, width=600, style={'background-color': '#F6F6F6'})))

    return pn.Column(*panels)