# app.py

from flask import Flask, request, render_template
from predict import run_prediction

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def home():
    plot1_html = None
    plot2_html = None

    if request.method == 'POST':
        stock_ticker = request.form.get('ticker')
        start_date = request.form.get('start_date')
        end_date = request.form.get('end_date')

        if stock_ticker and start_date and end_date:
            # This now returns HTML strings for the plots
            plot1_html, plot2_html = run_prediction(stock_ticker, start_date, end_date)

    return render_template(
        'index.html', 
        plot1_html=plot1_html, 
        plot2_html=plot2_html
    )

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
