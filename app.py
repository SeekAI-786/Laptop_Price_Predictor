import streamlit as st
import pickle
import numpy as np

# import the model
pipe = pickle.load(open(r'C:\laptop_price\pipe.pkl', 'rb'))
df = pickle.load(open(r'C:\laptop_price\df.pkl', 'rb'))

st.title("Laptop Predictor")

# brand
company = st.selectbox('Brand', df['Company'].unique())

# type of laptop
laptop_type = st.selectbox('Type', df['TypeName'].unique())

# Ram
ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

# weight
weight = st.number_input('Weight of the Laptop')

# Touchscreen
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

# IPS
ips = st.selectbox('IPS', ['No', 'Yes'])

# screen size
screen_size = st.number_input('Screen Size')

# resolution
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])

# cpu
cpu = st.selectbox('CPU', df['Cpu brand'].unique())

hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])

ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])

gpu = st.selectbox('GPU', df['Gpu brand'].unique())

os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    # query
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size
    query = np.array([str(company), str(laptop_type), str(ram), str(weight), str(touchscreen), str(ips), str(ppi), str(cpu), str(hdd), str(ssd), str(gpu), str(os)])

    st.write("Debugging Information:")
    st.write(f"Query Shape: {query.shape}")
    st.write(f"Query Content: {query}")

    try:
        query = query.reshape(1, -1)  # Ensure correct shape
        prediction = pipe.predict(query)

        # Assuming prediction is in log scale, convert it back to the original scale
        predicted_price = int(np.exp(prediction[0]))

        st.title("The predicted price of this configuration is $" + str(predicted_price))
    except Exception as e:
        st.error(f"An error occurred during prediction: {str(e)}")
