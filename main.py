import streamlit as st
import pandas as pd
from openai import OpenAI

from verbalizer.nlp import ParaphraseLanguageModel
from verbalizer.process import Processor
from verbalizer.sampler import Sampler
from verbalizer.verbalizer import Verbalizer, VerbalizerModelUsageConfig
from verbalizer.vocabulary import Vocabulary
from verbalizer.patterns import owl_disjoint, owl_restriction, owl_first_rest

patterns = [owl_disjoint.OwlDisjointWith, owl_restriction.OwlRestrictionPattern, owl_first_rest.OwlFirstRestPattern]


class LLM(ParaphraseLanguageModel):
    models = {
        "gpt-4o": {
            "input": 0.005,
            "output": 0.015
        },
        "gpt-4-0125-preview": {
            "input": 0.01,
            "output": 0.03
        },
        "gpt-4-1106-preview": {
            "input": 0.01,
            "output": 0.03
        },
        "gpt-4-1106-vision-preview": {
            "input": 0.01,
            "output": 0.03
        },
        "gpt-4": {
            "input": 0.03,
            "output": 0.06
        },
        "gpt-4-0613": {
            "input": 0.03,
            "output": 0.06
        },
        "gpt-4-32k": {
            "input": 0.06,
            "output": 0.12
        },
        "gpt-3.5-turbo-0125": {
            "input": 0.0005,
            "output": 0.0015
        },
        "gpt-3.5-turbo-instruct": {
            "input": 0.0015,
            "output": 0.0020
        },
        "gpt-3.5-turbo-16k-0613": {
            "input": 0.0030,
            "output": 0.0040
        },
        "gpt-3.5-turbo-1106": {
            "input": 0.0010,
            "output": 0.0020
        },
        "gpt-3.5-turbo-0613": {
            "input": 0.0015,
            "output": 0.0020
        },
        "gpt-3.5-turbo-0301": {
            "input": 0.0015,
            "output": 0.0020
        }
    }
    default_system_message = "You are an extremely specific data expert capable of converting pseudo English sentences into a meaningful and casual paragraph without losing information and integrate them into broader text in the form of articles and research papers. Avoid repeating information. Spell out everything don't be lazy!"
    default_prompt_template = [
        {
            "User": "sundried tomato topping disjoint with a sliced tomato topping.\n"
                    "sundried tomato topping sub class of (type a restriction, and on property a has spiciness, and some values from a mild).\n"
                    "sundried tomato topping sub class of a tomato topping.\n"
                    "sundried tomato topping type a class.",
            "Assistant": "The sundried tomato topping stands out as a unique entity, distinct from the sliced tomato topping, though both fall under the broader category of tomato toppings. What sets the sundried variety apart is its mild spiciness, a specific characteristic that enhances its versatility in culinary applications. Classified as a type of class, the sundried tomato topping is defined not only by its spiciness but also by its broader significance within an ontological framework. This structured classification helps in understanding its unique properties and potential uses, from recipe creation to organizing culinary data."
        }
    ]

    def __init__(self, model_name, temperature, message_templates, api_key):
        self.model_name = model_name
        self.temperature = temperature
        self.message_templates = message_templates
        self.client = OpenAI(api_key=api_key)
        self._in_token_usage = 0
        self._out_token_usage = 0

    def pseudo_to_text(self, pseudo_text: str, extra: str = None) -> str:
        messages = self.message_templates.copy()
        messages.append({"role": "user", "content": pseudo_text})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature
        )
        self._in_token_usage += response.usage.prompt_tokens
        self._out_token_usage += response.usage.completion_tokens
        return response.choices[0].message.content

    @property
    def cost(self) -> float:
        model_pricing = self.models.get(self.model_name) or {'input': 0.0, 'output': 0.0}

        in_tokens = self._in_token_usage / 1000
        out_tokens = self._out_token_usage / 1000

        return in_tokens * model_pricing['input'] + out_tokens * model_pricing["output"]

    @property
    def name(self) -> str:
        return self.model_name


@st.cache_data
def load_ontology(file) -> ('Graph', Vocabulary):
    bytes_data = file.read()
    graph = Processor.from_file(bytes_data)
    vocabulary = Vocabulary(graph)

    return graph, vocabulary


def vocabulary_from(vocabulary: Vocabulary, relationships_to_remove: pd.DataFrame, rephrased_identifiers: pd.DataFrame):
    return Vocabulary(
        graph=vocabulary.graph,
        ignore=set(relationships_to_remove["IRI"].tolist()),
        rephrased=rephrased_identifiers.set_index('IRI')['Display'].to_dict(),
    )


def replace_label(element: dict, vocabulary: Vocabulary):
    iri = element.get("root")
    element['root'] = vocabulary.get_class_label(iri)
    return element


def compute(
        initial_vocabulary: Vocabulary,
        relationships_to_remove: pd.DataFrame,
        rephrased_identifiers: pd.DataFrame,
        sampler: Sampler,
        llm: LLM
):
    print(llm.name)
    vocabulary = vocabulary_from(initial_vocabulary, relationships_to_remove, rephrased_identifiers)
    verbalizer = Verbalizer(
        vocabulary, patterns=patterns, language_model=llm, usage_config=VerbalizerModelUsageConfig(0, 1, "")
    )
    results = Processor.verbalize_with(verbalizer, namespace="main", sampler=sampler, as_generator=True)
    st.session_state.preview_results = results


# Streamlit application
def main():
    st.title("Ontology Verbalization Application")

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

        if sample_mode == "Fixed Size":
            sample_size = st.number_input(
                "Number of samples", min_value=10, max_value=max_concepts, value=max_concepts
            )
            sampler = Sampler(sample_n=sample_size, seed=st.number_input("Seed", value=42))
        else:
            sample_percentage = st.slider("Sample Percentage", min_value=10, max_value=100, value=100) / 100
            sampler = Sampler(sample_percentage=sample_percentage, seed=st.number_input("Seed", value=42))

        st.success("Sampling configuration set.")

    # Step 4: Prompt Configurations
    st.header("Step 4: Prompt Configurations (Optional)", help="Recommended to generate quality text",
              divider=True)
    llm = None
    if st.toggle("Enable LLM"):
        model_name = st.selectbox("Model to Use", list(LLM.models.keys()))
        api_key = st.text_input("OpenAI API Key", type="password")
        system_prompt = st.text_area(
            "System Prompt",
            value=LLM.default_system_message,
            placeholder="Enter the system prompt here"
        )

        st.subheader("Few-Shot Examples")
        few_shots = st.data_editor(
            pd.DataFrame(LLM.default_prompt_template, columns=["User", "Assistant"]),
            use_container_width=True,
            num_rows="dynamic"
        )

        temperature = st.slider("Temperature", min_value=0.0, max_value=2.0, value=1.0)

        message_templates = [
            {"role": "system", "content": system_prompt}
        ]
        for i, row in few_shots.iterrows():
            if row["User"] and row["Assistant"]:
                message_templates.append({"role": "user", "content": row["User"]})
                message_templates.append({"role": "assistant", "content": row["Assistant"]})

        if model_name and api_key:
            llm = LLM(
                model_name=model_name,
                temperature=temperature,
                message_templates=message_templates,
                api_key=api_key
            )

    st.subheader("Actions")
    preview_sampler = Sampler(sample_n=10, seed=42)

    if 'preview_results' not in st.session_state:
        st.session_state.preview_results = None

    st.button(
        "Preview", on_click=compute,
        args=(initial_vocabulary, relationships_to_remove, rephrased_identifiers, preview_sampler, llm)
    )

    preview_columns = ["root", "fragment", "text", "llm_text"]

    if st.session_state.preview_results:
        try:
            first = next(st.session_state.preview_results)
            preview_table = st.dataframe(
                pd.DataFrame([replace_label(first, initial_vocabulary)], columns=preview_columns))
            for result in st.session_state.preview_results:
                result = {key: value for key, value in result.items() if key in preview_columns}
                preview_table.add_rows(
                    pd.DataFrame([replace_label(result, initial_vocabulary)], columns=preview_columns))
        except:
            pass


if __name__ == "__main__":
    main()
