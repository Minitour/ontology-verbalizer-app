import pandas as pd
import streamlit as st
from verbalizer.patterns import owl_disjoint, owl_restriction, owl_first_rest
from verbalizer.process import Processor
from verbalizer.sampler import Sampler
from verbalizer.verbalizer import Verbalizer, VerbalizerModelUsageConfig, VerbalizationInitError
from verbalizer.vocabulary import Vocabulary
from app import utils

patterns = [owl_disjoint.OwlDisjointWith, owl_restriction.OwlRestrictionPattern, owl_first_rest.OwlFirstRestPattern]


@st.cache_data
def load_ontology(file) -> ('Graph', Vocabulary):
    bytes_data = file.read()
    graph = Processor.from_file(bytes_data)
    vocabulary = Vocabulary(graph)

    return graph, vocabulary


def compute(
        initial_vocabulary: Vocabulary,
        relationships_to_remove: pd.DataFrame,
        rephrased_identifiers: pd.DataFrame,
        sampler: Sampler,
        llm: utils.LLM
):
    try:
        vocabulary = utils.vocabulary_from(initial_vocabulary, relationships_to_remove, rephrased_identifiers)
        verbalizer = Verbalizer(
            vocabulary, patterns=patterns, language_model=llm, usage_config=VerbalizerModelUsageConfig(0, 1, "")
        )
        results = Processor.verbalize_with(verbalizer, namespace="main", sampler=sampler, as_generator=True)
        st.session_state.preview_results = results
        st.session_state.preview_error = None
    except VerbalizationInitError as e:
        st.session_state.preview_error = f'Error: {e}'

def render_results_preview(vocabulary: Vocabulary):
    if not st.session_state.preview_results:
        return

    preview_columns = ["root", "fragment", "text", "llm_text"]

    if isinstance(st.session_state.preview_results, list):
        st.dataframe(pd.DataFrame(st.session_state.preview_results, columns=preview_columns))
        return

    try:
        first = next(st.session_state.preview_results)
        first = utils.replace_label(first, vocabulary)
        new_preview_results = []
        new_preview_results.append(first)
        preview_table = st.dataframe(pd.DataFrame([first], columns=preview_columns))
        for result in st.session_state.preview_results:
            result = {key: value for key, value in result.items() if key in preview_columns}
            result = utils.replace_label(result, vocabulary)
            new_preview_results.append(result)
            preview_table.add_rows(pd.DataFrame([result], columns=preview_columns))

        st.session_state.preview_results = new_preview_results
    except Exception as e:
        print(e)
        st.session_state.preview_results = []
        st.error('Something went wrong during verbalization. Try to adjust your configurations.')


# Streamlit application
def main():
    st.set_page_config(page_title="Ontology Verbalization App", page_icon=":material/edit_note:")
    st.title("Ontology Verbalization App")

    # Step 1: Upload Ontology OWL File
    st.header("Step 1: Upload Ontology OWL File", divider=True)
    uploaded_file = st.file_uploader("Drag and drop or select an OWL file", type=["owl", "ttl"])

    if not uploaded_file:
        return

    graph, initial_vocabulary = load_ontology(uploaded_file)
    st.success("Ontology loaded successfully!")

    # Step 2: Select Relationships to Keep
    st.header("Step 2: Select Relationships to Keep", divider=True)
    relationships = pd.DataFrame.from_dict(
        initial_vocabulary.relationship_labels, orient="index", columns=["Label"]
    ).reset_index().rename(columns={"index": "IRI"})
    relationships["Keep"] = False

    edited_relationships = st.data_editor(
        relationships, use_container_width=True, num_rows="dynamic"
    )
    selected_relationships = edited_relationships[edited_relationships["Keep"]]
    relationships_to_remove = edited_relationships[~edited_relationships["Keep"]]

    st.subheader("Override Identifier Names")
    rephrased_identifiers = st.data_editor(
        pd.DataFrame(
            [
                {"IRI": key, 'Display': value}
                for key, value in
                {
                    'http://www.w3.org/2002/07/owl#equivalentClass': 'is same as',
                    'http://www.w3.org/2000/01/rdf-schema#subClassOf': 'is a type of',
                    'http://www.w3.org/2002/07/owl#intersectionOf': 'all of',
                    'http://www.w3.org/2002/07/owl#unionOf': 'any of',
                    'http://www.w3.org/2002/07/owl#disjointWith': 'is different from',
                    'http://www.w3.org/2002/07/owl#withRestrictions': 'must be',
                    'http://purl.obolibrary.org/obo/IAO_0000115': 'has definition',
                }.items()
            ]
        ),
        use_container_width=True,
        num_rows="dynamic"
    )

    if len(selected_relationships) == 0:
        st.warning("At least one relationship must be selected to proceed.")
        return

    st.success(f"{len(selected_relationships)} relationships selected.")

    # Step 3: Select Sample Size
    st.header("Step 3: Select Sample Size (Optional)", help="Recommended for large datasets", divider=True)
    sampler = None
    if st.toggle("Enable Sampling"):
        max_concepts = len(initial_vocabulary.object_labels)
        sample_mode = st.radio("Sampling Mode", ["Fixed Size", "Percentage"], index=0)
        col1, col2 = st.columns([1, 1])
        if sample_mode == "Fixed Size":
            with col1:
                sample_size = st.number_input(
                    "Number of samples", min_value=10, max_value=max_concepts, value=max_concepts
                )
            with col2:
                seed = st.number_input("Seed", value=42)
            sampler = Sampler(sample_n=sample_size, seed=seed)
        else:
            with col1:
                sample_percentage = st.slider("Sample Percentage", min_value=10, max_value=100, value=100) / 100
            with col2:
                seed = st.number_input("Seed", value=42)
            sampler = Sampler(sample_percentage=sample_percentage, seed=seed)

        st.success("Sampling configuration set.")

    # Step 4: Prompt Configurations
    st.header("Step 4: Prompt Configurations (Optional)", help="Recommended to generate quality text",
              divider=True)
    llm = None
    if st.toggle("Enable LLM"):
        model_name = st.selectbox("Model to Use", list(utils.LLM.models.keys()))
        api_key = st.text_input("OpenAI API Key", type="password")
        system_prompt = st.text_area(
            "System Prompt",
            value=utils.LLM.default_system_message,
            placeholder="Enter the system prompt here"
        )

        st.subheader("Few-Shot Examples")
        few_shots = st.data_editor(
            pd.DataFrame(utils.LLM.default_prompt_template, columns=["User", "Assistant"]),
            use_container_width=True,
            num_rows="dynamic"
        )

        temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=0.7, step=0.1)

        message_templates = [
            {"role": "system", "content": system_prompt}
        ]
        for i, row in few_shots.iterrows():
            if row["User"] and row["Assistant"]:
                message_templates.append({"role": "user", "content": row["User"]})
                message_templates.append({"role": "assistant", "content": row["Assistant"]})

        if model_name and api_key:
            llm = utils.LLM(
                model_name=model_name,
                temperature=temperature,
                message_templates=message_templates,
                api_key=api_key
            )
            st.success("LLM configured successfully!")

    st.subheader("Actions")
    preview_sampler = Sampler(sample_n=10, seed=42)

    if 'preview_results' not in st.session_state:
        st.session_state.preview_results = None

    if 'preview_error' not in st.session_state:
        st.session_state.preview_error = None

    col1, col2 = st.columns([1, 1])
    with col1:
        st.button(
            "Preview Verbalization",
            on_click=compute,
            args=(initial_vocabulary, relationships_to_remove, rephrased_identifiers, preview_sampler, llm),
            use_container_width=True,
            help="Verbalize up to 10 samples"
        )
    with col2:
        st.button(
            "Verbalize",
            on_click=compute,
            args=(initial_vocabulary, relationships_to_remove, rephrased_identifiers, sampler, llm),
            use_container_width=True,
            help=f"Perform verbalization of {utils.get_number_of_concepts(initial_vocabulary, sampler)} concepts"
        )

    code = utils.generate_code(uploaded_file.name, relationships_to_remove, rephrased_identifiers, sampler, llm)

    if st.session_state.preview_error:
        st.error(st.session_state.preview_error)

    with st.expander("Show Code Snippet"):
        st.code(code, language="python")

    render_results_preview(initial_vocabulary)


if __name__ == "__main__":
    main()
