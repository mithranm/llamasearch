# Online sources and Gen AI has been used to help with adapting
# the code and fixing minor mistakes
import gradio as gr
from gradio import Blocks
from utils import save_to_db, QARecord, export_to_txt, delete_all_records


def generate_response(link, user_input):
    return f"AI response for: {user_input} (Link: {link})"


def download_voice():
    return "voice_file.mp3"


def download_chat():
    export_to_txt("conversation.txt")


def rate_response(rating):
    print(f"You rated: {rating}")


def submit_review(review):
    return f"Review submitted: {review}"

<<<<<<< Updated upstream
=======
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
    
    # Clear question and answer; for the radio, try setting to an empty string
    return gr.update(value=""), gr.update(value=""), gr.update(value=None)

def new_chat():
    """
    Deletes all records from the database and clears all UI fields,
    including link input, question, chat output, rating, and review components.
    """
    print("new_chat() called")
    delete_all_records()
    return (
        gr.update(value=""),                 # Clear link_input
        gr.update(value=""),                 # Clear user_input (question)
        gr.update(value=""),                 # Clear chat_output (answer)
        gr.update(value=""),                 # Clear rating (set to empty string)
        gr.update(value="", visible=False),  # Clear and hide review_prompt
        gr.update(visible=False)             # Hide submit_review_btn
    )


>>>>>>> Stashed changes

def create_app() -> Blocks:
    with gr.Blocks() as demo:
        gr.Markdown("# LLAMASEARCH - Next Gen AI Search Assistant")

<<<<<<< Updated upstream
        link_input = gr.Textbox(
            label="Enter a Website Link", placeholder="https://example.com"
        )

        chat_output = gr.Textbox(label="AI Response", interactive=False, lines=15)

        with gr.Row():
            user_input = gr.Textbox(
                label="Ask a Question", placeholder="Type your question here..."
            )
            submit_btn = gr.Button("Search")
=======
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

        
>>>>>>> Stashed changes

        with gr.Row():
            voice_btn = gr.Button("Download Voice")
            chat_btn = gr.Button("Download Chat History")
            rating = gr.Radio(["", "👍", "👎"], label="Rate Response", value="")
            review_btn = gr.Button("Leave a Review")

        review_prompt = gr.Textbox(
            label="Your Review", visible=False, placeholder="Write your review here..."
        )
        submit_review_btn = gr.Button("Submit Review", visible=False)

        submit_btn.click(
            generate_response, inputs=[link_input, user_input], outputs=chat_output
        )
        voice_btn.click(download_voice, outputs=None)
        chat_btn.click(download_chat, outputs=None)
        rating.change(rate_response, inputs=rating, outputs=None)
        
        new_chat_btn.click(
            new_chat, 
            inputs=[], 
            outputs=[link_input, user_input, chat_output, rating, review_prompt, submit_review_btn]
        )


        next_question_btn.click(
            save_and_clear, 
            inputs=[user_input, chat_output, rating], 
            outputs=[user_input, chat_output, rating]
        )

        def show_review_prompt():
            return gr.update(visible=True), gr.update(visible=True)

        review_btn.click(show_review_prompt, outputs=[review_prompt, submit_review_btn])
        submit_review_btn.click(submit_review, inputs=review_prompt, outputs=None)
    return demo
<<<<<<< Updated upstream
=======

if __name__ == "__main__":
    app = create_app()
    app.launch()
>>>>>>> Stashed changes
