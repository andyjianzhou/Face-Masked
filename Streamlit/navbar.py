import streamlit as st

def navbar():
    
    st.markdown("""
    <style>
        body {
            font-family: 'Trebuchet MS', sans-serif;
            font-weight: bold;
            margin: 0;
            background-color: whitesmoke;
        } 
        .block-container {
            padding: 0px !important;
            margin: 0px auto !important;
            width: 100% !important;
            }
        /* #31333f */
        ul {
            list-style-type: none;
            font-size: large;
            margin: 0;
            padding: 0;
            overflow: hidden;
            background-color: #333;
            position: relative;
            width: 100%;
            height: 3.5rem;
            z-index: 999999;
        }
        li {
            height: 100%;
            float: left;
        }
        li a {
            display: block;
            color: white;
            text-align: center;
            padding: 18px 16px;
            height: 100%;
            text-decoration: none;
        }

        li a:hover:not(.active) {
            background-color: #111;
        }

        .active {
            background-color: #333;
        }

        .logo-image {
            width: 46px;
            height: 46px;
            overflow: hidden;
            margin-top: -25%;
        }

        iframe{
            width: 100%;
            border: none;
            position: fixed;
            height: 100%;
            z-index: 0;
        }
        .dropbtn {
            background-color: #333;
            color: rgba(255, 255, 255, 0.356);
            padding: 16px;
            font-size: large;
            font-family: 'Trebuchet MS', sans-serif;
            font-weight: bold;
            border: none;
            cursor: pointer;
            height: 3.5rem;

        }

        .dropdown {
        position: fixed;
        right: 0;
        z-index: 9999999999999;
        }

        .dropdown-content {
        display: none;
        position: absolute;
        background-color: #f9f9f9;
        min-width: 160px;
        box-shadow: 0px 8px 16px 0px rgba(0,0,0,0.2);
        z-index: 9999999;
        }

        .dropdown-content a {
        color: black;
        padding: 12px 16px;
        text-decoration: none;
        display: block;
        }

        .dropdown-content a:hover {
            background-color: #f1f1f1
        }

        .dropdown:hover .dropdown-content {
        display: block;
        }

        .dropdown:hover .dropbtn {
            background-color: #333;
            color: rgba(255, 255, 255, 0.726);
        }
        .navbtn a {
            background-color: #333;
            color: rgba(255, 255, 255, 0.356);
            padding: 12px 16px;
            text-decoration: none;
            display: block;
        }
        .navbtn a:hover {
            color: rgba(255, 255, 255, 0.726);
        }
    </style>
        """, unsafe_allow_html=True)


    st.markdown("""
    <body>
        <div class="dropdown">
            <button class="dropbtn">Dropdown</button>
            <div class="dropdown-content">
                <a href="#">Link 1</a>
                <a href="#">Link 2</a>
                <a href="#">Link 3</a>
            </div>
        </div>
        <ul>
            <li>
                <a class="navbar-brand" href="/">
                    <div class="logo-image">
                    <img src="/static/images/logo.png" class="img-fluid">
                    </div>
                </a>
            </li>
            <div class="navbtn">
                <li>
                    <a href="/?p=About" onclick="setURL('localhost:8501/?p=About')">About</a>
                </li>
                <li>
                    <a href="/?p=YouTube" onclick="setURL('localhost:8501/?p=YouTube')">YouTube</a>
                </li>
                <li>
                    <a href="/?p=Demos" onclick="setURL('localhost:8501/?p=examples')">Demos</a>
                </li>
            </div>
        </ul>
    </body>
    """, unsafe_allow_html=True)