from flask import Flask, render_template,request
import pickle
import numpy as np
popular_df = pickle.load(open('Book_recommender_project(3)/popular.pkl','rb'))
books = pickle.load(open('Book_recommender_project(3)/books.pkl','rb'))
pt = pickle.load(open('Book_recommender_project(3)/pt.pkl','rb'))
Similarity = pickle.load(open('Book_recommender_project(3)/Similarity.pkl','rb'))

app=Flask(__name__)

@app.route("/")
def index(): 
    return render_template("index.html",
                           book_name = list(popular_df["Book-Title"].values),
                           author =list(popular_df["Book-Author"].values),
                           image =list(popular_df["Image-URL-M"].values),
                           votes =list(popular_df["num_ratings"].values),
                           rating= list(popular_df["avg_rating"].values),
                           )
@app.route('/recommend')
def recommend_ui(): 
    return render_template("Recommend.html")

@app.route("/recommend_books", methods=["POST", "GET"])
def recommend():
    user_input = request.form["user_input"]

    # Check if the user_input exists in the index
    if user_input not in pt.index:
        return render_template("Recommend.html", error="Book not found in recommendations.")

    # Retrieve the index of the user input
    index = np.where(pt.index == user_input)[0][0]
    Similar_items = sorted(list(enumerate(Similarity[index])), key=lambda x: x[1], reverse=True)[1:6]
    data = []
    
    for i in Similar_items:
        item = []
        temp_df = books[books["Book-Title"] == pt.index[i[0]]]
        item.extend(list(temp_df.drop_duplicates("Book-Title")["Book-Title"].values))
        item.extend(list(temp_df.drop_duplicates("Book-Title")["Book-Author"].values))
        item.extend(list(temp_df.drop_duplicates("Book-Title")["Image-URL-M"].values))
        data.append(item)

    return render_template("Recommend.html", data=data)


if __name__ == "__main__": 
    app.run(debug=True , port=5000)