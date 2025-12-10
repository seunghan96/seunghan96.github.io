```
import pandas as pd


def check_similar_users(user_id_x, user_id_y, ratings):
    if user_id_x == user_id_y:
        return False

    joint_ratings = pd.merge(
        ratings[ratings["user id"] == user_id_x],
        ratings[ratings["user id"] == user_id_y],
        on="item id",
    )
    if joint_ratings.empty:
        return False

    joint_ratings["rating_diff"] = abs(
        joint_ratings["rating_x"] - joint_ratings["rating_y"]
    )
    if max(joint_ratings["rating_diff"]) <= 1:
        return True
    return False


def get_recommendations(users, movies, ratings, full_name, method, year):
    # (1) Validation
    check_user = full_name in users["full name"].values
    check_year = year in movies["release year"].values
    check_method = method in ["by_popularity", "by_rating", "by_similar_users"]

    if not (check_user * check_year * check_method):
        return ""

    # (2) Extract rows
    user_id = users.loc[users["full name"] == full_name, "user id"].iloc[0]
    item_ids = ratings.loc[ratings["user id"] == user_id, "item id"]
    movies_year = movies[movies["release year"] == year]
    cands = movies_year[~movies_year["movie id"].isin(item_ids)]
    if len(cands) == 0:
        return ""
    
    # (3) Method 1 & 2: Popularity & Ratings
    if (method == "by_popularity") or (method == "by_rating"):
        cands_ratings = ratings[ratings["item id"].isin(cands["movie id"])]
        if len(cands_ratings) == 0:
            return ""
        if method == "by_popularity":
            rating_stat = cands_ratings.groupby("item id")["rating"].count()
        else:
            rating_stat = cands_ratings.groupby("item id")["rating"].mean()
        idx_max = rating_stat[rating_stat == rating_stat.max()].index
        titles = movies[movies["movie id"].isin(idx_max)]["movie title"]
        return sorted(titles)[0]
    
    # (4) Method 3: Similar users
    elif method == "by_similar_users":
        similar_users = [id for id in users["user id"] if ((id != user_id) and (check_similar_users(user_id, id, ratings)))]
        if not similar_users:
            return ""
        
        cands_ratings = ratings[
            (ratings["user id"].isin(similar_users))
            & (ratings["item id"].isin(cands["movie id"]))
        ]
        if len(cands_ratings)==0:
            return ""
        
        latest_by_item = cands_ratings.groupby("item id")["timestamp"].max()
        latest_max = latest_by_item.max()
        top_ids = latest_by_item[latest_by_item == latest_max].index
        titles = movies.loc[movies["movie id"].isin(top_ids), "movie title"].tolist()
        return sorted(titles)[0] 
    return ""
```

