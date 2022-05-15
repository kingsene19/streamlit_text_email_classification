# Load Database Pkg
import sqlite3
conn = sqlite3.connect('data.db')
c = conn.cursor()


# Fxn
def create_page_visited_table():
	c.execute('CREATE TABLE IF NOT EXISTS pageTrackTable(pagename TEXT,dateDeVisite TIMESTAMP)')

def add_page_visited_details(pagename,dateDeVisite):
	c.execute('INSERT INTO pageTrackTable(pagename,dateDeVisite) VALUES(?,?)',(pagename,dateDeVisite))
	conn.commit()

def view_all_page_visited_details():
	c.execute('SELECT * FROM pageTrackTable')
	data = c.fetchall()
	return data


# Fxn To Track Input & Prediction
def create_emotionclf_table():
	c.execute('CREATE TABLE IF NOT EXISTS emotionclfTable(text TEXT,prediction TEXT,probabilite NUMBER,dateDeVisite TIMESTAMP)')

def add_prediction_details(rawtext,prediction,probability,timeOfvisit):
	c.execute('INSERT INTO emotionclfTable(text,prediction,probabilite,dateDeVisite) VALUES(?,?,?,?)',(rawtext,prediction,probability,timeOfvisit))
	conn.commit()

def view_all_prediction_details():
	c.execute('SELECT * FROM emotionclfTable')
	data = c.fetchall()
	return data