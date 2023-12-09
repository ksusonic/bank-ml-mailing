import time
import streamlit as st
import pandas as pd

from PIL import Image

from model import open_data, preprocess_data, get_importances, predict_on_input


def preload_content():
    _, _, _, _, scaler = preprocess_data(open_data('dataset.csv'))

    background = Image.open('static/bank.jpg')
    age = Image.open('static/age.png')
    heatmap = Image.open('static/heatmap.png')
    income = Image.open('static/income.png')
    income_with_target = Image.open('static/income_with_target.png')

    return scaler, background, age, heatmap, income, income_with_target


def highlight_weighs(s):
    return ['background-color: #E6F6E4'] * len(s) if s['Вес'] > 0 else ['background-color: #F6EBE4'] * len(s)


def pack_input(sex, age, child, dependants, work, pens, income, loans, closed_loans):
    return pd.DataFrame({
        'AGE': age,
        'GENDER': 1 if sex == 'Мужской' else 0,
        'CHILD_TOTAL': child,
        'DEPENDANTS': dependants,
        'SOCSTATUS_WORK_FL': 1 if work == 'Трудоустроен' else 0,
        'SOCSTATUS_PENS_FL': 1 if pens == 'На пенсии' else 0,
        'PERSONAL_INCOME': income,
        'LOAN_NUM_TOTAL': loans,
        'LOAN_NUM_CLOSED': closed_loans,
    }, index=[0])


def render_page(scaler, background, age, heatmap, income, income_with_target):
    st.title('Рассылка банковских предложений')
    st.subheader('Исследуем релевантность для клиентов по различным признакам, предсказываем успешность показа')
    st.write(
        'Материал - данные клиентов банка и target, означающий заинтересовало ли клиента предложение (понятие "клик")'
    )
    st.image(background)

    tab1, tab2, tab3 = st.tabs([':mag: Исследовать', ':mage: Предсказать', ':vertical_traffic_light: Оценить'])

    with tab1:
        st.write('Exploratory data analysis: исследуем наши данные, предварительно очищенные и обработанные :sparkles:')

        st.write('**Возраст клиентов**')
        st.image(age)
        st.write('Самый распространённый возраст - от 20 до 60 лет')
        st.divider()

        st.write('**Корреляция признаков**')
        st.image(heatmap)
        st.write('Клиентов примерно равно распределены по полу')
        st.write(
            'Наиболее коррелирующие признаки: пенсия и возраст (логично, тк пенсия появляется с определенного возраста)'
            ', иждивенцы и дети. На target наиболее влияет возраст и доход.')
        st.divider()

        st.write('**Распределение выборки по доходу:**')
        st.image(income)
        st.write('Выборка выглядит естественно, но содержатся выбросы в слишком высокий доход.')
        st.divider()

        st.write('**Влияние дохода на клик по показу:**')
        st.image(income_with_target)
        st.write('У людей с более высоким доходом клики происходят чаще')
        st.divider()

    with tab2:
        st.write('Введите данные клиента:')

        col1, col2, col3 = st.columns(3)
        with col1:
            sex = st.selectbox('Пол', ['Женский', 'Мужской'])
            age = st.slider('Возраст', min_value=0, max_value=100)
        with col2:
            child = st.slider('Количество детей', min_value=0, max_value=100)
            dependants = st.slider('Количество иждивенцев', min_value=0, max_value=100)
        with col3:
            work = st.selectbox('Статус работы', ['Трудоустроен', 'Безработный'])
            pens = st.selectbox('Пенсия', ['Нет', 'На пенсии'])
        col1, col2 = st.columns(2)
        with col1:
            income = st.slider('Среднемесячный личный доход', min_value=0, max_value=1000000)
            loans = st.slider('Взято кредитов:', min_value=0, max_value=10)
            closed_loans = st.slider('Возвращено кредитов:', min_value=0, max_value=loans) if loans > 0 else 0
        st.divider()

        col1, col2, col3 = st.columns(3)
        if col2.button('Предсказать :mage:'):
            with st.spinner('Считаем!'):
                time.sleep(1)
                inputs = pack_input(sex, age, child, dependants, work, pens, income, loans, closed_loans)
                scaled = pd.DataFrame(scaler.transform(inputs), columns=inputs.columns)

                pred, proba = predict_on_input(scaled)
                if pred == 1:
                    st.success('Вас заинтересует наше предложение! :thumbsup: :thumbsup:')
                    with st.expander('Подробнее'):
                        st.write(f'Вероятность этого: **`{round(max(proba[0]), 3)}`**')
                elif pred == 0:
                    st.error('Боюсь, вы не заинтересуетесь предложением :thumbsdown: :thumbsdown:')
                    with st.expander('Подробнее'):
                        st.write(f'Вероятность этого: **`{round(max(proba[0]), 3)}`**')
                else:
                    st.error('Что-то пошло не так...')

    with tab3:
        col1, col2 = st.columns(2)
        with col1:
            st.write('**Что важно для клика клиента?**')
            st.dataframe(get_importances(5, 'most').style.apply(highlight_weighs, axis=1))
        with col2:
            st.write('**А что практически не важно?**')
            st.dataframe(get_importances(5, 'least').style.apply(highlight_weighs, axis=1))


def load_page():
    st.set_page_config(layout="wide",
                       page_title="Удолетворенность рассылкой",
                       page_icon='🦒')
    scaler, background, age, heatmap, income, income_with_target = preload_content()
    render_page(scaler, background, age, heatmap, income, income_with_target)


if __name__ == "__main__":
    load_page()
