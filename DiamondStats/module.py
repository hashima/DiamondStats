import glob
import os
import numpy as np
import cv2
import pandas as pd
pd.set_option('display.max_columns', None)
import seaborn as sns
sns.set()
import matplotlib.pyplot as plt
import japanize_matplotlib
japanize_matplotlib.japanize()
import base64

def loadWorkSheet(gc, id, sheetname):

    url = "https://docs.google.com/spreadsheets/d/{}/".format(id)
    ss = gc.open_by_url(url)

    # シートを特定する（シート名で特定）
    st = ss.worksheet(sheetname)

    # スプレッドシートのデータをPandasのDataframeに変換する。
    rows = st.get_all_values()
    df_master = pd.DataFrame.from_records(rows[1:], columns=rows[0])

    return df_master

def initDataFrames(df_master, df_gameinfo, df_score, dt_from, dt_to, target_game, throws):

    df_master = df_master.loc[:, :'投手左右']
    df_gameinfo = df_gameinfo.loc[:, :'対戦相手']
    df_gameinfo = df_gameinfo.dropna(subset=['試合No'])
    df_gameinfo['試合No'] = df_gameinfo['試合No'].dropna().astype(int)
    df_gameinfo['日付'] = pd.to_datetime(df_gameinfo['日付'])
    if target_game == '公式戦':
        df_gameinfo = df_gameinfo[~df_gameinfo['試合種別'].isin(['A戦 第1試合', 'A戦 第2試合', 'B戦 第1試合', 'B戦 第2試合'])]
    elif target_game == 'OP戦':
        df_gameinfo = df_gameinfo[df_gameinfo['試合種別'].isin(['A戦 第1試合', 'A戦 第2試合', 'B戦 第1試合', 'B戦 第2試合'])]
    elif target_game == 'A戦':
        df_gameinfo = df_gameinfo[df_gameinfo['試合種別'].isin(['A戦 第1試合', 'A戦 第2試合'])]
    elif target_game == 'B戦':
        df_gameinfo = df_gameinfo[df_gameinfo['試合種別'].isin(['B戦 第1試合', 'B戦 第2試合'])]

    if dt_from != "" and dt_to != "":
        df_gameinfo = df_gameinfo[pd.to_datetime(dt_from) <= df_gameinfo['日付'] and df_gameinfo['日付'] <= pd.to_datetime(dt_to)]
    elif dt_from != "":
        df_gameinfo = df_gameinfo[pd.to_datetime(dt_from) <= df_gameinfo['日付']]
    elif dt_to != "":
        df_gameinfo = df_gameinfo[df_gameinfo['日付'] <= pd.to_datetime(dt_to)]

    if not dt_from == None or not dt_to == None:
        df_score = df_score[df_score['試合No'].isin(df_gameinfo['試合No'])]
    
    df_score = df_score.loc[:, :'アウト']
    df_score = df_score.dropna(subset=['試合No'])
    if throws == '対右投手':
        df_score = df_score.query("投 == '右'").copy()
    if throws == '対左投手':
        df_score = df_score.query("投 == '左'").copy()
    df_score['試合No'] = df_score['試合No'].dropna().astype(int)
    df_score['打席'] = df_score['打席'].dropna().astype(int)
    df_score['アウト'] = df_score['アウト'].dropna().astype(int)
    df_score['B'] = 0
    df_score['S'] = 0
    df_score['B'] = df_score['B'].dropna().astype(int)
    df_score['S'] = df_score['S'].dropna().astype(int)

    df_score = setBSO(df_score)

    return df_master, df_gameinfo, df_score

def setBSO(df_score):

    game_no = 0
    ab_no = 0
    b = 0
    s = 0
    for row in df_score.itertuples(index=True, name="Row"):
        if row.試合No != game_no or row.打席 != ab_no:
            b = 0
            s = 0
            game_no = row.試合No
            ab_no = row.打席

        if row.投球結果.startswith("(S)") or row.投球結果.startswith("(F)"):
            s = min(s + 1, 2)
        if row.投球結果.startswith("(B)") or row.投球結果.startswith("(F)"):
            b = min(b + 1, 3)

        df_score.at[row.Index, "B"] = b
        df_score.at[row.Index, "S"] = s

    return df_score 

def getScoreResult(df_score):

    df_score_result_all = df_score[df_score['打撃結果'].str.len() > 0]
    target_results = ['安打', '二塁打', '三塁打', '本塁打', '本塁打(R)', 'エンタイトル二塁打']
    df_score_result_hit = df_score_result_all[df_score_result_all['打撃結果'].isin(target_results)]

    return df_score_result_all, df_score_result_hit

def getCourseResult(df_master, df_score_result_ab, df_score_result_hit):

    df_course_ab = df_score_result_ab.groupby('コース')['打席'].count().reset_index()
    df_course_hit = df_score_result_hit.groupby('コース')['打席'].count().reset_index()
    df_course_k = df_score_result_ab[df_score_result_ab['打撃結果'] == '三振'].groupby('コース')['打撃結果'].count().reset_index()

    df_course = df_master[['コース']].dropna(subset=['コース'])
    df_course = pd.merge(df_course, df_course_ab, on='コース', how='left')
    df_course = pd.merge(df_course, df_course_hit, on='コース', how='left')
    df_course = pd.merge(df_course, df_course_k, on='コース', how='left')
    df_course = df_course.rename(columns={'打席_x': '打数'})
    df_course = df_course.rename(columns={'打席_y': '安打数'})
    df_course = df_course.rename(columns={'打撃結果': '三振数'})
    df_course = df_course.fillna(0)
    df_course['打率'] = df_course['安打数'] / df_course['打数'].replace(0, float('nan'))
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
    ave = target_rows / all_rows if all_rows > 0 else 0
    # print(f"打率: {round(ave, 3)}")

    return ave

def drawCourse(img, x, y, _df_course):

    course = [['(B)上1', '(B)上2',    '(B)上3',    '(B)上4',   '(B)上5'],
              ['(B)左1', '(S)左上',   '(S)中央上', '(S)右上',   '(B)右1'],
              ['(B)左2', '(S)左中央', '(S)中央',   '(S)右中央', '(B)右2'],
              ['(B)左3', '(S)左下',   '(S)中央下', '(S)右下',   '(B)右3'],
              ['(B)下1', '(B)下2',    '(B)下3',    '(B)下4',   '(B)下5']]

    select = course[x][y]
    # 打数
    ab = int(_df_course[_df_course['コース'] == select]['打数'].iloc[0])
    # 打率
    ave = float(_df_course[_df_course['コース'] == select]['打率'].iloc[0])
    # 安打数
    hit = int(_df_course[_df_course['コース'] == select]['安打数'].iloc[0])
    # 三振数
    k = int(_df_course[_df_course['コース'] == select]['三振数'].iloc[0])
    cv2.rectangle(img, (5+x*92, 5+y*133), (5+(x+1)*92, 5+(y+1)*133), (0, 0, 0))
    if ab > 0:
        (text_width, text_height), baseline = cv2.getTextSize(f"{ab}-{hit}", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(img, f"{ab}-{hit}", (5+x*92+(92-text_width)//2, 5+y*133+40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), thickness=2)
        (text_width, text_height), baseline = cv2.getTextSize(f"{ave:.3f}".lstrip('0'), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(img, f"{ave:.3f}".lstrip('0'), (5+x*92+(92-text_width)//2, 5+y*133+70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), thickness=2)
        (text_width, text_height), baseline = cv2.getTextSize(f"K : {k}", cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.putText(img, f"K : {k}", (5+x*92+(92-text_width)//2, 5+y*133+100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), thickness=2)

def getCountAB(df_score_result_ab, df_score_result_hit):

    df_count_ab = pd.DataFrame(columns=["カウント", "打率", "打数", "安打", "単打", "2塁打", "3塁打", "本塁打"])

    for b in range(0,4):
        for s in range(0,3):
            df_match_ab = df_score_result_ab.query("B == {} and S == {}".format(b, s))

            if df_match_ab.empty:
                df_count_ab.loc[len(df_count_ab)] = ["{} - {}".format(b, s), 0, 0, 0, 0, 0, 0, 0]
            else:
                df_match_hit = df_score_result_hit.query("B == {} and S == {}".format(b, s))
                df_match_single = df_score_result_hit.query("B == {} and S == {} and 打撃結果 == '安打'".format(b, s))
                df_match_2h = df_score_result_hit.query("B == {} and S == {} and (打撃結果 == '二塁打' or 打撃結果 == 'エンタイトル二塁打')".format(b, s))
                df_match_3h = df_score_result_hit.query("B == {} and S == {} and 打撃結果 == '三塁打'".format(b, s))
                df_match_hr = df_score_result_hit.query("B == {} and S == {} and (打撃結果 == '本塁打' or 打撃結果 == '本塁打(R)')".format(b, s))
                df_count_ab.loc[len(df_count_ab)] = ["{} - {}".format(b, s), len(df_match_hit) / len(df_match_ab), len(df_match_ab), len(df_match_hit), len(df_match_single), len(df_match_2h), len(df_match_3h), len(df_match_hr)]

    df_count_ab['打率'] = df_count_ab['打率'].map(lambda x: f"{x:.3f}".lstrip('0'))

    return df_count_ab
    
def getDirectionResult(df_master, df_score_result_ab, df_score_result_hit):
    df_direction_ab = df_score_result_ab.groupby('打球方向')['打席'].count().reset_index()
    df_direction_ab = df_direction_ab.rename(columns={'打席': '打数'})
    df_direction_hit = df_score_result_hit.groupby('打球方向')['打席'].count().reset_index()
    df_direction_hit = df_direction_hit.rename(columns={'打席': '安打数'})

    df_direction = df_master[['打球方向']].dropna(subset=['打球方向'])
    df_direction = pd.merge(df_direction, df_direction_ab, on='打球方向', how='left')
    df_direction = pd.merge(df_direction, df_direction_hit, on='打球方向', how='left')
    df_direction['打数'] = df_direction['打数'].fillna(0).astype(int)
    df_direction['安打数'] = df_direction['安打数'].fillna(0).astype(int)

    list_direction_ad = [0,0,0,0,0,0,0,0,0]
    list_direction_hit = [0,0,0,0,0,0,0,0,0]
    list_direction_ad[0] = int(df_direction[df_direction['打球方向'] == '投手']['打数'].iloc[0])
    list_direction_hit[0] = int(df_direction[df_direction['打球方向'] == '投手']['安打数'].iloc[0])
    list_direction_ad[1] = int(df_direction[df_direction['打球方向'] == '捕手']['打数'].iloc[0])
    list_direction_hit[1] = int(df_direction[df_direction['打球方向'] == '捕手']['安打数'].iloc[0])
    list_direction_ad[2] = int(df_direction[df_direction['打球方向'] == '一塁手']['打数'].iloc[0])
    list_direction_hit[2] = int(df_direction[df_direction['打球方向'] == '一塁手']['安打数'].iloc[0])
    list_direction_ad[3] = int(df_direction[df_direction['打球方向'] == '二塁手']['打数'].iloc[0])
    list_direction_hit[3] = int(df_direction[df_direction['打球方向'] == '二塁手']['安打数'].iloc[0])
    list_direction_ad[4] = int(df_direction[df_direction['打球方向'] == '三塁手']['打数'].iloc[0])
    list_direction_hit[4] = int(df_direction[df_direction['打球方向'] == '三塁手']['安打数'].iloc[0])
    list_direction_ad[5] = int(df_direction[df_direction['打球方向'] == '遊撃手']['打数'].iloc[0])
    list_direction_hit[5] = int(df_direction[df_direction['打球方向'] == '遊撃手']['安打数'].iloc[0])
    list_direction_ad[6] = int(df_direction[df_direction['打球方向'] == '左翼手']['打数'].iloc[0]) + int(df_direction[df_direction['打球方向'] == '三遊間']['打数'].iloc[0]) + int(df_direction[df_direction['打球方向'] == 'レフト線']['打数'].iloc[0]) + int(df_direction[df_direction['打球方向'] == 'レフトオーバー']['打数'].iloc[0])
    list_direction_hit[6] = int(df_direction[df_direction['打球方向'] == '左翼手']['安打数'].iloc[0]) + int(df_direction[df_direction['打球方向'] == '三遊間']['安打数'].iloc[0]) + int(df_direction[df_direction['打球方向'] == 'レフト線']['安打数'].iloc[0]) + int(df_direction[df_direction['打球方向'] == 'レフトオーバー']['安打数'].iloc[0])
    list_direction_ad[7] = int(df_direction[df_direction['打球方向'] == '中堅手']['打数'].iloc[0]) + int(df_direction[df_direction['打球方向'] == '左中間']['打数'].iloc[0]) + int(df_direction[df_direction['打球方向'] == '右中間']['打数'].iloc[0]) + int(df_direction[df_direction['打球方向'] == 'センターオーバー']['打数'].iloc[0])
    list_direction_hit[7] = int(df_direction[df_direction['打球方向'] == '中堅手']['安打数'].iloc[0]) + int(df_direction[df_direction['打球方向'] == '左中間']['安打数'].iloc[0]) + int(df_direction[df_direction['打球方向'] == '右中間']['安打数'].iloc[0]) + int(df_direction[df_direction['打球方向'] == 'センターオーバー']['安打数'].iloc[0])
    list_direction_ad[8] = int(df_direction[df_direction['打球方向'] == '右翼手']['打数'].iloc[0]) + int(df_direction[df_direction['打球方向'] == '一二塁間']['打数'].iloc[0]) + int(df_direction[df_direction['打球方向'] == 'ライト線']['打数'].iloc[0]) + int(df_direction[df_direction['打球方向'] == 'ライトオーバー']['打数'].iloc[0])
    list_direction_hit[8] = int(df_direction[df_direction['打球方向'] == '右翼手']['安打数'].iloc[0]) + int(df_direction[df_direction['打球方向'] == '一二塁間']['安打数'].iloc[0]) + int(df_direction[df_direction['打球方向'] == 'ライト線']['安打数'].iloc[0]) + int(df_direction[df_direction['打球方向'] == 'ライトオーバー']['安打数'].iloc[0])
    return list_direction_ad, list_direction_hit

def drawDirection(list_direction_ad, list_direction_hit):
    # キャンバス作成（白背景）
    height, width = 600, 800
    field = np.ones((height, width, 3), dtype=np.uint8) * 255

    # 中心座標とフィールドサイズ設定
    center_x, center_y = width // 2, height - 50
    radius = 300
    base_offset = 100

    # 外野の半円（緑：線画なので色は省略して黒に）
    cv2.ellipse(field, (center_x, center_y - 280), (radius, 250), 0, 184, 356, (0, 0, 0), 2)

    # 内野のダイヤモンド（四角形）
    pts = np.array([
        [center_x, center_y],                            # ホームベース
        [center_x - base_offset, center_y - base_offset],# サード
        [center_x, center_y - base_offset * 2],          # セカンド
        [center_x + base_offset, center_y - base_offset],# ファースト
    ], np.int32)
    # cv2.polylines(field, [pts], isClosed=True, color=(0, 0, 0), thickness=2)

    line_offset = 300
    cv2.line(field, (center_x, center_y), (center_x + line_offset, center_y - line_offset), (0, 0, 0), 2)
    cv2.line(field, (center_x, center_y), (center_x - line_offset, center_y - line_offset), (0, 0, 0), 2)

    # ピッチャーマウンド（小さい円）
    # cv2.circle(field, (center_x, center_y - int(base_offset * 1.2)), 10, (0, 0, 0), 2)

    # ベース（小さな四角で描画）
    # base_size = 7
    # for x, y in pts:
    #     cv2.rectangle(field, (x - base_size, y - base_size), (x + base_size, y + base_size), (0, 0, 0), 1)

    # ホームベースも描画
    # cv2.rectangle(field, (center_x - base_size, center_y - base_size), (center_x + base_size, center_y + base_size), (0, 0, 0), 1)

    ave = 0.333
    ab = 10
    hit = 3
    # ピッチャー
    ave = 0 if list_direction_ad[0] == 0 else list_direction_hit[0]/list_direction_ad[0]
    (text_width, text_height), baseline = cv2.getTextSize(f"{ave:.3f}".lstrip('0'), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    (ab_text_width, text_height), baseline = cv2.getTextSize(f"{list_direction_ad[0]}-{list_direction_hit[0]}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.putText(field, f"{ave:.3f}".lstrip('0'), (center_x - text_width//2, 420), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), thickness=2)
    cv2.putText(field, f"{list_direction_ad[0]}-{list_direction_hit[0]}", (center_x - ab_text_width//2, 443), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=2)
    # キャッチャー
    ave = 0 if list_direction_ad[1] == 0 else list_direction_hit[1]/list_direction_ad[1]
    (text_width, text_height), baseline = cv2.getTextSize(f"{ave:.3f}".lstrip('0'), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    (ab_text_width, text_height), baseline = cv2.getTextSize(f"{list_direction_ad[1]}-{list_direction_hit[1]}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.putText(field, f"{ave:.3f}".lstrip('0'), (center_x - text_width//2, 500), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), thickness=2)
    cv2.putText(field, f"{list_direction_ad[1]}-{list_direction_hit[1]}", (center_x - ab_text_width//2, 523), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=2)
    # ファースト
    ave = 0 if list_direction_ad[2] == 0 else list_direction_hit[2]/list_direction_ad[2]
    (text_width, text_height), baseline = cv2.getTextSize(f"{ave:.3f}".lstrip('0'), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    (ab_text_width, text_height), baseline = cv2.getTextSize(f"{list_direction_ad[2]}-{list_direction_hit[2]}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.putText(field, f"{ave:.3f}".lstrip('0'), (530 - text_width//2, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), thickness=2)
    cv2.putText(field, f"{list_direction_ad[2]}-{list_direction_hit[2]}", (530 - ab_text_width//2, 383), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=2)
    # セカンド
    ave = 0 if list_direction_ad[3] == 0 else list_direction_hit[3]/list_direction_ad[3]
    (text_width, text_height), baseline = cv2.getTextSize(f"{ave:.3f}".lstrip('0'), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    (ab_text_width, text_height), baseline = cv2.getTextSize(f"{list_direction_ad[3]}-{list_direction_hit[3]}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.putText(field, f"{ave:.3f}".lstrip('0'), (460 - text_width//2, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), thickness=2)
    cv2.putText(field, f"{list_direction_ad[3]}-{list_direction_hit[3]}", (460 - ab_text_width//2, 323), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=2)
    # サード
    ave = 0 if list_direction_ad[4] == 0 else list_direction_hit[4]/list_direction_ad[4]
    (text_width, text_height), baseline = cv2.getTextSize(f"{ave:.3f}".lstrip('0'), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    (ab_text_width, text_height), baseline = cv2.getTextSize(f"{list_direction_ad[4]}-{list_direction_hit[4]}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.putText(field, f"{ave:.3f}".lstrip('0'), (270 - text_width//2, 360), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), thickness=2)
    cv2.putText(field, f"{list_direction_ad[4]}-{list_direction_hit[4]}", (270 - ab_text_width//2, 383), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=2)
    # ショート
    ave = 0 if list_direction_ad[5] == 0 else list_direction_hit[5]/list_direction_ad[5]
    (text_width, text_height), baseline = cv2.getTextSize(f"{ave:.3f}".lstrip('0'), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    (ab_text_width, text_height), baseline = cv2.getTextSize(f"{list_direction_ad[5]}-{list_direction_hit[5]}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.putText(field, f"{ave:.3f}".lstrip('0'), (340 - text_width//2, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), thickness=2)
    cv2.putText(field, f"{list_direction_ad[5]}-{list_direction_hit[5]}", (340 - ab_text_width//2, 323), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=2)
    # レフト
    ave = 0 if list_direction_ad[6] == 0 else list_direction_hit[6]/list_direction_ad[6]
    (text_width, text_height), baseline = cv2.getTextSize(f"{ave:.3f}".lstrip('0'), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    (ab_text_width, text_height), baseline = cv2.getTextSize(f"{list_direction_ad[6]}-{list_direction_hit[6]}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.putText(field, f"{ave:.3f}".lstrip('0'), (240 - text_width//2, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), thickness=2)
    cv2.putText(field, f"{list_direction_ad[6]}-{list_direction_hit[6]}", (240 - ab_text_width//2, 233), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=2)
    # センター
    ave = 0 if list_direction_ad[7] == 0 else list_direction_hit[7]/list_direction_ad[7]
    (text_width, text_height), baseline = cv2.getTextSize(f"{ave:.3f}".lstrip('0'), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    (ab_text_width, text_height), baseline = cv2.getTextSize(f"{list_direction_ad[7]}-{list_direction_hit[7]}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.putText(field, f"{ave:.3f}".lstrip('0'), (center_x - text_width//2, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), thickness=2)
    cv2.putText(field, f"{list_direction_ad[7]}-{list_direction_hit[7]}", (center_x - ab_text_width//2, 143), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=2)
    # ライト
    ave = 0 if list_direction_ad[8] == 0 else list_direction_hit[8]/list_direction_ad[8]
    (text_width, text_height), baseline = cv2.getTextSize(f"{ave:.3f}".lstrip('0'), cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    (ab_text_width, text_height), baseline = cv2.getTextSize(f"{list_direction_ad[8]}-{list_direction_hit[8]}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
    cv2.putText(field, f"{ave:.3f}".lstrip('0'), (560 - text_width//2, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), thickness=2)
    cv2.putText(field, f"{list_direction_ad[8]}-{list_direction_hit[8]}", (560 - ab_text_width//2, 233), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), thickness=2)

    return field