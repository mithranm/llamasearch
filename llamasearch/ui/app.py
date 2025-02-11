# Online sources and Gen AI has been used to help with adapting 
# the code and fixing minor mistakes
import gradio as gr

def generate_response(link, user_input):
    return f"AI response for: {user_input} (Link: {link})"

def download_voice():
    return "voice_file.mp3"

def download_chat():
    return "chat_history.txt"

def rate_response(rating):
    return f"You rated: {rating}"

def submit_review(review):
    return f"Review submitted: {review}"

with gr.Blocks() as demo:
    gr.Markdown("# LLAMASEARCH - Next Gen AI Search Assistant")

    link_input = gr.Textbox(label="Enter a Website Link", 
                            placeholder="https://example.com")
    
    chat_output = gr.Textbox(label="AI Response", interactive=False, lines=15)

    with gr.Row():
        user_input = gr.Textbox(label="Ask a Question", 
                                placeholder="Type your question here...")
        submit_btn = gr.Button("Search")

    with gr.Row():
        voice_btn = gr.Button("Download Voice")
        chat_btn = gr.Button("Download Chat History")
        rating = gr.Radio(["👍", "👎"], label="Rate Response")
        review_btn = gr.Button("Leave a Review")

    review_prompt = gr.Textbox(label="Your Review", visible=False, 
                               placeholder="Write your review here...")
    submit_review_btn = gr.Button("Submit Review", visible=False)

    submit_btn.click(generate_response, inputs=[link_input, user_input], 
                     outputs=chat_output)
    voice_btn.click(download_voice, outputs=None)
    chat_btn.click(download_chat, outputs=None)
    rating.change(rate_response, inputs=rating, outputs=None)

    def show_review_prompt():
        return gr.update(visible=True), gr.update(visible=True)

    review_btn.click(show_review_prompt, outputs=[review_prompt, submit_review_btn])
    submit_review_btn.click(submit_review, inputs=review_prompt, outputs=None)

demo.launch()
