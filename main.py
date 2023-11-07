from test2 import WebsiteChatApp
import json
import openai
import constants
def get_individual_answer(question):
    obj = WebsiteChatApp()
    response = obj.get_website_resp(question)
    return response

def get_summarized_answer(question):
    response = get_individual_answer(question)
    res_str = '\n\n'.join(list(response.values()))
    print(res_str)
    gpt_prompt = f"Act as a content Summarizer and give the best summarized answer for" \
                 f"by taking reference from {res_str}. Make sure output would be properly" \
                 f"formatted and meaningfull. If answer is not found, dont add anything by " \
                 f"your own, just return as not found"
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=gpt_prompt,
            temperature=0.6,
            max_tokens=150,
            n=1,
            stop=None
        )

        summary = response.choices[0].text.strip()
        print(summary)
        return summary
    except Exception as e:
        print(e)




def save_websites_to_session(website_name):
    """Save websites to the session file."""
    session_file_path = "websites_session.json"
    with open(session_file_path, "r") as f:
        data = json.load(f)
    data.append(website_name)

    with open(session_file_path, "w") as f:
        json.dump(data, f)

