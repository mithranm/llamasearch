# Online sources and Gen AI has been used to help with adapting
# the code and fixing minor mistakes
import os
import shutil
import base64 
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

def download_chat() -> str:
    """
    Exports chat history into a uniquely‑named file in ~/Downloads:
    conversation.txt, conversation(1).txt, conversation(2).txt, etc.
    Returns the full path so you can display it in the UI.
    """
    # 1) Ensure the Downloads folder exists
    downloads_dir = os.path.join(os.path.expanduser("~"), "Downloads")
    os.makedirs(downloads_dir, exist_ok=True)

    # 2) Build a base name and extension
    base = "conversation"
    ext = ".txt"

    # 3) Find the first available filename
    dest_path = os.path.join(downloads_dir, base + ext)
    counter = 1
    while os.path.exists(dest_path):
        dest_path = os.path.join(downloads_dir, f"{base}({counter}){ext}")
        counter += 1

    # 4) Export to that path
    export_to_txt(dest_path)


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
        gr.update(value=None),
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
        gr.update(visible=False),
    )

def upload_files(file_paths):
    """
    Copies each selected file to llamasearch/temp.
    Returns a status string to display in the UI.
    """
    if not file_paths:
        return "No files selected."

    # Ensure the temp folder exists
    os.makedirs("llamasearch/temp", exist_ok=True)

    # file_paths will be a list of strings if file_count="multiple" and type="filepath"
    uploaded = []
    for path in file_paths:
        # Extract the original filename from the path
        filename = os.path.basename(path)
        destination = os.path.join("llamasearch", "temp", filename)
        shutil.copy(path, destination)
        uploaded.append(filename)

    return f"Uploaded: {', '.join(uploaded)}"

def create_app() -> Blocks:
    with gr.Blocks(title="LlamaSearch") as demo:
        icon_path = "temp/llamasearch.png"  # update if needed
        if os.path.isfile(icon_path):
            with open(icon_path, "rb") as f:
                base64_data = base64.b64encode(f.read()).decode("utf-8")
            # Create an <img> tag with a data URI
            icon_html = f"""
            <div style="display: flex; align-items: center;">
              <img 
                src="data:image/png;base64,{base64_data}"
                alt="LlamaSearch Icon" 
                style="width:80px; height:auto; margin-right:15px;"/>
              <h2 style="margin:0;">LlamaSearch</h2>
            </div>
            """
        else:
            # Fallback: file not found
            icon_html = "<h2 style='color:red;'>Icon not found!</h2>"

        with gr.Row():
            gr.HTML(icon_html)

        with gr.Row():
            # --- Column 1 ---
            with gr.Column():
                link_input = gr.Textbox(
                    label="Enter a Website Link",
                    placeholder="https://example.com"
                )
            
            # --- Column 2 ---
            with gr.Column():
                file_input = gr.File(
                    label="Attach Files",
                    file_count="multiple",
                    type="filepath"
                )
            
            # --- Column 3 ---
            with gr.Column():
                upload_btn = gr.Button("Upload")
                upload_status = gr.Textbox(label="Status", interactive=False)
        
        # Wire up the upload button
        upload_btn.click(
            fn=upload_files,
            inputs=file_input,
            outputs=upload_status
        )

        user_input = gr.Textbox(
            label="Ask a Question",
            placeholder="Type your question here...",
            elem_id="wider-textbox",
        )
        
        with gr.Row():
            submit_btn = gr.Button("Search")
            next_question_btn = gr.Button("Next Question")
            new_chat_btn = gr.Button("New Chat")

        with gr.Row():
            with gr.Column(scale=3):
                # Large column for AI response
                chat_output = gr.Textbox(label="AI Response", interactive=False, lines=15)
                # Audio output can be under chat

            with gr.Column(scale=1):
                # Stack all buttons on the right
                listen_btn = gr.Button("Listen Answer")
                audio_output = gr.Audio(label="Audio Response", interactive=False)

                # download_status = gr.Textbox(label="Download Status", interactive=False)
                download_btn = gr.Button("Download Chat History")
                download_btn.click(
                    fn=download_chat,
                    inputs=None
                )

                rating = gr.Radio(["", "👍", "👎"], label="Rate Response", value="")
                review_btn = gr.Button("Leave a Review")

        review_prompt = gr.Textbox(
            label="Your Review", visible=False, placeholder="Write your review here..."
        )
        submit_review_btn = gr.Button("Submit Review", visible=False)

        submit_btn.click(
            generate_response, inputs=[link_input, user_input], outputs=chat_output
        )
        listen_btn.click(listen_answer, inputs=chat_output, outputs=audio_output)

        rating.change(rate_response, inputs=rating, outputs=None)

        new_chat_btn.click(
            new_chat,
            inputs=[],
            outputs=[
                link_input,
                user_input,
                chat_output,
                rating,
                review_prompt,
                submit_review_btn,
                audio_output,
            ],
        )

        next_question_btn.click(
            save_and_clear,
            inputs=[user_input, chat_output, rating],
            outputs=[user_input, chat_output, rating, audio_output],
        )

        def show_review_prompt():
            return gr.update(visible=True), gr.update(visible=True)

        review_btn.click(show_review_prompt, outputs=[review_prompt, submit_review_btn])
        submit_review_btn.click(submit_review, inputs=review_prompt, outputs=None)
    return demo


if __name__ == "__main__":
    app = create_app()
    app.launch(favicon_path="temp/llamasearch.png")