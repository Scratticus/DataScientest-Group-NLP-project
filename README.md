## Explanations and instructions

This repository contains the files needed to initialize a fil-rouge project as part of your [DataScientest](https://datascientest.com/) training program.

It mainly contains the present README.md file and an application template [Streamlit](https://streamlit.io/).

**README**

The README.md file is a central element of any git repository. It is used to introduce your project and its objectives, as well as to explain how to install and launch the project, or even contribute to it.

You'll need to modify various sections of this README.md to include the necessary information.

- Complete **in English** the sections (`## Presentation` and `## Installation` `## Streamlit App`) by following the instructions in these sections.
- Delete this section (`## Explanations and Instructions`)

**Streamlit application

A [Streamlit](https://streamlit.io/) application template is available in the [`streamlit_app`](streamlit_app) folder. You can use this template to showcase your project.

## Presentation

Complete this section **in English** with a brief description of your project, background (including a link to the DataScientest course), and objectives.

You can also add a brief presentation of the team members with links to your respective networks (GitHub and/or LinkedIn for example).

**Example:**

This repository contains the code for our **PROJECT_NAME** project, developed during our [Data Scientist training](https://datascientest.com/en/data-scientist-course) at [DataScientest](https://datascientest.com/).

The aim of this project is to **...**

This project was developed by the following team:

- John Doe ([GitHub](https://github.com/) / [LinkedIn](http://linkedin.com/))
- Martin Dupont ([GitHub](https://github.com/) / [LinkedIn](http://linkedin.com/))

You can browse and run [notebooks](./notebooks). You will need to install the dependencies (in a dedicated environment):

```
pip install -r requirements.txt
```

## Streamlit App

**Add explanations on how to use the application **.

To launch the application :

``shell
cd streamlit_app
conda create --name my-awesome-streamlit python=3.9
conda activate my-awesome-streamlit
pip install -r requirements.txt
streamlit run app.py
```

The application should now be available on [localhost:8501](http://localhost:8501).
