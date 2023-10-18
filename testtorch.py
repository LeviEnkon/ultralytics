import pandas as pd
import datetime
import time
# Initialize empty DataFrame
detects = pd.DataFrame(columns=["id", "newy", "newd", "newxan", "last_accessed"])

def access_or_add(id, newy=None, newd=None, newxan=None):
    global detects
    now = datetime.datetime.now()
    # Check if id exists
    if id in detects['id'].values:
        detects.loc[detects['id'] == id, 'last_accessed'] = now
        detects.loc[detects['id'] == id, 'newy'] = newy
        detects.loc[detects['id'] == id, 'newd'] = newd
        detects.loc[detects['id'] == id, 'newxan'] = newxan
    else:
        # Add new row with the provided details
        detects = detects._append({"id": id, "newy": newy, "newd": newd, "newxan":newxan, "last_accessed": now}, ignore_index=True)

def cleanup_old_entries():
    global detects
    now = datetime.datetime.now()
    thirty_seconds_ago = now - datetime.timedelta(seconds=5)
    # Only keep rows accessed within the last 30 seconds
    detects = detects[detects['last_accessed'] > thirty_seconds_ago]

id=1
ydown=480
dist=100
angle_x=20
access_or_add(id, ydown, dist, angle_x)
print(detects)

lasty=detects.loc[detects['id'] == id, 'newy'].values[0]
lastd=detects.loc[detects['id'] == id, 'newd'].values[0]
lastxan=detects.loc[detects['id'] == id, 'newxan'].values[0]
print(lasty, lastd, lastxan)
id=1
ydown=420
dist=80
angle_x=15
access_or_add(id, ydown, dist, angle_x)
print(detects)
time.sleep(3)
cleanup_old_entries()
print("3seconds")
print(detects)
time.sleep(3)
cleanup_old_entries()
print("6seconds")
print(detects)