import streamlit as st

from pathlib import Path

from  text_processor import TextCleaner
import joblib
import time


# page configuration 
st.set_page_config(page_title="IMDB AI Sentiment Analysis", page_icon="🎦")


st.title(" 🎬 AI Movies Review Sentiment Analysis", text_alignment="center")


input_text  = st.text_area(label="## write you impression on the movie here", placeholder="The cinematography was brilliant, but the plot was lacking...")

# instantiate the TextCleaner class
test_cleaner = TextCleaner()

# load the pipeline function
@st.cache_resource
def load_pipeline(model_path):
    return joblib.load(model_path)

# load the pipeline

model_path = Path(__file__).parent.parent / "models" / "final_50_000_LR_pipeline_v1.joblib"
pipeline = load_pipeline(model_path)

# add a button 
if st.button(label="Analyze review sentiment", type="primary" , shortcut="Enter"):
    result_container = st.empty()
    result_container.empty()
    
    if input_text:
        with st.spinner("In progress...", show_time=True):
            time.sleep(2)
            #### model logic
            # clean the text
            cleaned_text = test_cleaner.clean_text(input_text)
            # make prediction with the model
            prediction = pipeline.predict([cleaned_text])
            predict_proba = pipeline.predict_proba([cleaned_text])

        with result_container.container():
            st.divider()

            if prediction  == 1 :
                success_message =   st.success(f"### POSITIVE:  with a score of {predict_proba[0][1]:.2f}%")
                st.balloons()
            else:
                failure_message =  st.error(f"### NEGATIVE:  with a score of {predict_proba[0][0]:.2f}%")
                
            st.subheader("💡key influencers")
            st.write("model focused on these words:")
            st.info(f"_{cleaned_text}_")
    else:
        st.warning("Please enter a review first")
