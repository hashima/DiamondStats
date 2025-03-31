import pandas as pd
import glob
import os
import numpy as np
import cv2


def loadWorkSheet(gc, id, sheetname):

    url = "https://docs.google.com/spreadsheets/d/{}/".format(id)
    ss = gc.open_by_url(url)

    # シートを特定する（シート名で特定）
    st = ss.worksheet(sheetname)

    # スプレッドシートのデータをPandasのDataframeに変換する。
    rows = st.get_all_values()
    df_master = pd.DataFrame.from_records(rows[1:], columns=rows[0])

    return df_master

def initDataFrames(df_master, df_gameinfo, df_score):

    df_master = df_master.loc[:, :'投手左右']
    df_gameinfo = df_gameinfo.loc[:, :'対戦相手']
    df_gameinfo = df_gameinfo.dropna(subset=['試合No'])
    df_gameinfo['試合No'] = df_gameinfo['試合No'].dropna().astype(int)
    df_score = df_score.loc[:, :'打撃結果']
    df_score = df_score.dropna(subset=['試合No'])
    df_score['試合No'] = df_score['試合No'].dropna().astype(int)
    df_score['打席'] = df_score['打席'].dropna().astype(int)

    return df_master, df_gameinfo, df_score

def getScoreResult(df_score):

    df_score_result_all = df_score[df_score['打撃結果'].str.len() > 0]
    target_results = ['安打', '二塁打', '三塁打', '本塁打', '本塁打(R)', 'エンタイトル二塁打']
    df_score_result_hit = df_score_result_all[df_score_result_all['打撃結果'].isin(target_results)]

    return df_score_result_all, df_score_result_hit

def getCourseResult(df_master, df_score_result_ab, df_score_result_hit):

    df_course_ab = df_score_result_ab.groupby('コース')['打席'].count().reset_index()
    df_course_hit = df_score_result_hit.groupby('コース')['打席'].count().reset_index()

    df_course = df_master[['コース']].dropna(subset=['コース'])
    df_course = pd.merge(df_course, df_course_ab, on='コース', how='left')
    df_course = pd.merge(df_course, df_course_hit, on='コース', how='left')
    df_course = df_course.fillna(0)
    df_course['ave'] = df_course['打席_y'] / df_course['打席_x'].replace(0, float('nan'))
    df_course = df_course.fillna(0)

    return df_course

def getCourseResultImg(df_course):

    height = 675
    width = 471
    img = np.ones((height, width, 3)) * 255

    [drawCourse(img, x, y, df_course) for x in range(0, 5) for y in range(0, 5)]
    cv2.rectangle(img, (97, 138), (373, 537), (0, 0, 0), thickness=2)

    return img

def getItems(df_master):

    game_cat_list = df_master['試合種別'].dropna().to_list()
    ground_list = df_master['球場'].dropna().to_list()
    opponent_list = df_master['対戦相手'].dropna().to_list()
    course_list = df_master['コース'].dropna().to_list()
    pitch_list = df_master['球種'].dropna().to_list()
    pitching_esults_list = df_master['投球結果'].dropna().to_list()
    ball_attributes_list = df_master['打球属性'].dropna().to_list()
    ball_direction_list = df_master['打球方向'].dropna().to_list()
    batting_results_list = df_master['打撃結果'].dropna().to_list()
    pitcher_RL_list = df_master['投手左右'].dropna().to_list()

    return game_cat_list, ground_list, opponent_list, course_list, pitch_list, pitching_esults_list, ball_attributes_list, ball_direction_list, batting_results_list, pitcher_RL_list

def getBattingAverage(df_score):

    df_score_result = df_score[df_score['打撃結果'].str.len() > 0]

    target_results = ['安打', '二塁打', '三塁打', '本塁打', '本塁打(R)', 'エンタイトル二塁打']
    target_rows = len(df_score_result[df_score_result['打撃結果'].isin(target_results)])

    all_rows = len(df_score_result)
    # 結果の表示
    ave = target_rows / all_rows
    # print(f"打率: {round(ave, 3)}")

    return ave



def drawCourse(img, x, y, _df_course):

    course = [['(B)上1', '(B)上2',    '(B)上3',    '(B)上4',   '(B)上5'],
              ['(B)左1', '(S)左上',   '(S)中央上', '(S)右上',   '(B)右1'],
              ['(B)左2', '(S)左中央', '(S)中央',   '(S)右中央', '(B)右2'],
              ['(B)左3', '(S)左下',   '(S)中央下', '(S)右下',   '(B)右3'],
              ['(B)下1', '(B)下2',    '(B)下3',    '(B)下4',   '(B)下5']]

    select = course[x][y]
    ab = float(_df_course[_df_course['コース'] == select]['打席_x'].iloc[0])
    ave = float(_df_course[_df_course['コース'] == select]['ave'].iloc[0])
    hit = float(_df_course[_df_course['コース'] == select]['打席_y'].iloc[0])
    cv2.rectangle(img, (5+x*92, 5+y*133), (5+(x+1)*92, 5+(y+1)*133), (0, 0, 0))
    cv2.putText(img, f"{ave:.3f}", (5+x*92+10, 5+y*133+60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)
