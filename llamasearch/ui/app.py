# Online sources and Gen AI has been used to help with adapting
# the code and fixing minor mistakes
import gradio as gr
from gradio import Blocks
from llamasearch.ui.utils import save_to_db, QARecord, export_to_txt, delete_all_records

def generate_response(link, user_input):
    return f"AI response for: {user_input} (Link: {link})"

# def download_voice():
#    return "voice_file.mp3"
def listen_answer(answer_text):
    """
    Converts the provided answer text to speech using gTTS,
    saves it as a temporary MP3 file, and returns the file path.
    """
    from gtts import gTTS
    import tempfile
    if not answer_text.strip():
        return None
    tts = gTTS(answer_text)
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tmp_file.close()  # Close the file so gTTS can write to it.
    tts.save(tmp_file.name)
    return tmp_file.name

def download_chat():
    export_to_txt("conversation.txt")

def rate_response(rating):
    print(f"You rated: {rating}")

def submit_review(review):
    return f"Review submitted: {review}"

def save_and_clear(question, answer, rating):
    """
    This function creates a QARecord from the inputs, saves it to the database,
    and then clears the question, answer, and rating fields.
    """
    # Convert rating from emoji to integer (default is 0)
    if rating == "👍":
        rating_int = 1
    elif rating == "👎":
        rating_int = -1
    else:
        rating_int = 0

    # Create the QARecord and save it to the database
    record = QARecord(question=question, answer=answer, rating=rating_int)
    save_to_db(record)
    
    return (
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(visible=False)
    )

def new_chat():
    """
    Deletes all records from the database and clears all UI fields,
    including link input, question, chat output, rating, and review components.
    """
    print("new_chat() called")
    delete_all_records()
    return (
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value=""),
        gr.update(value="", visible=False),
        gr.update(visible=False),
        gr.update(visible=False)
    )

def create_app() -> Blocks:
    with gr.Blocks() as demo:
        gr.Markdown("# LLAMASEARCH - Next Gen AI Search Assistant")

        link_input = gr.Textbox(label="Enter a Website Link", 
                                placeholder="https://example.com")
        
        user_input = gr.Textbox(
                label="Ask a Question", 
                placeholder="Type your question here...", 
                elem_id="wider-textbox"
            )
        
        with gr.Row():
            submit_btn = gr.Button("Search")
            next_question_btn = gr.Button("Next Question")
            new_chat_btn = gr.Button("New Chat")

        chat_output = gr.Textbox(label="AI Response", interactive=False, lines=15)

        with gr.Row():
            listen_btn = gr.Button("Listen Answer")
            chat_btn = gr.Button("Download Chat History")
            rating = gr.Radio(["", "👍", "👎"], label="Rate Response", value="")
            review_btn = gr.Button("Leave a Review")

        audio_output = gr.Audio(label="Audio Response", interactive=False)

        review_prompt = gr.Textbox(label="Your Review", visible=False, 
                                   placeholder="Write your review here...")
        submit_review_btn = gr.Button("Submit Review", visible=False)

        submit_btn.click(generate_response, inputs=[link_input, user_input], 
                         outputs=chat_output)
        listen_btn.click(
            listen_answer, 
            inputs=chat_output, 
            outputs=audio_output
        )
        chat_btn.click(download_chat, outputs=None)
        rating.change(rate_response, inputs=rating, outputs=None)
        
        new_chat_btn.click(
            new_chat, 
            inputs=[], 
            outputs=[link_input, user_input, chat_output, rating, review_prompt, submit_review_btn, audio_output]
        )

        next_question_btn.click(
            save_and_clear, 
            inputs=[user_input, chat_output, rating], 
            outputs=[user_input, chat_output, rating, audio_output]
        )

        def show_review_prompt():
            return gr.update(visible=True), gr.update(visible=True)

        review_btn.click(show_review_prompt, outputs=[review_prompt, submit_review_btn])
        submit_review_btn.click(submit_review, inputs=review_prompt, outputs=None)
    return demo

if __name__ == "__main__":
    app = create_app()
    app.launch()
    