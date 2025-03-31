import argparse
import os
import pandas as pd
import cv2
import numpy as np

import DiamondStats
import DiamondStats.module as ds

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
    cv2.putText(img, str(round(ave, 3)), (5+x*92+10, 5+y*133+60), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), thickness=2)

    
if __name__ == "__main__":

    # マスタデータ読み込み
    # df_master = ds.loadWorkSheet('1zwas-6Xf5UgTnX_Hcx_DvB0xVV-ugqI55QAz9N6cDqU', '入力規則')
    # df_gameinfo = ds.loadWorkSheet('1zwas-6Xf5UgTnX_Hcx_DvB0xVV-ugqI55QAz9N6cDqU', '試合情報')
    # df_score = ds.loadWorkSheet('1zwas-6Xf5UgTnX_Hcx_DvB0xVV-ugqI55QAz9N6cDqU', '打席スコア')
    df_master = pd.read_csv('DiamondStats/data/Master.csv')
    df_gameinfo = pd.read_csv('DiamondStats/data/gameinfo.csv')
    df_score = pd.read_csv('DiamondStats/data/score.csv')
 
    # データフレームの初期化
    df_master, df_gameinfo, df_score = ds.initDataFrames(df_master, df_gameinfo, df_score)

    # 各アイテムのマスタを取得
    game_cat_list, ground_list, opponent_list, course_list, pitch_list, pitching_esults_list, ball_attributes_list, ball_direction_list, batting_results_list, pitcher_RL_list = ds.getItems(df_master)

    # 通算打率の取得
    ave = ds.getBattingAverage(df_score)
    print(f"打率: {ave:.3f}")

    # df_score = pd.merge(df_score, df_gameinfo, on='試合No', how='left')

    # 打席と安打打席の取得
    df_score_result_ab, df_score_result_hit = ds.getScoreResult(df_score)

    # コース別の結果の取得
    df_course = ds.getCourseResult(df_master, df_score_result_ab, df_score_result_hit)

    # コース別の結果画像の取得
    img = ds.getCourseResultImg(df_course)


    # cv2_imshow(img)
    cv2.imshow('Image-1', img)
    # キー入力を待つ
    cv2.waitKey(0)
    # キーが入力されたら画像を閉じる
    cv2.destroyAllWindows()

    # # フィールドサイズ
    # width, height = 800, 800

    # # 白色の背景画像を作成
    # field = np.ones((height, width, 3), dtype=np.uint8) * 255

    # # フィールドの中心座標
    # center = (width // 2, height // 2)

    # # 野球ダイヤモンド型フィールドを描く
    # # 内野を描画（四角形）
    # cv2.rectangle(field, (center[0] - 200, center[1] - 200), (center[0] + 200, center[1] + 200), (0, 255, 0), 2)

    # # 3つのベースを描く（ホーム、ファースト、セカンド）
    # cv2.circle(field, (center[0] - 200, center[1]), 10, (0, 0, 255), -1)  # ファースト
    # cv2.circle(field, (center[0] + 200, center[1]), 10, (0, 0, 255), -1)  # セカンド
    # cv2.circle(field, (center[0], center[1] + 200), 10, (0, 0, 255), -1)  # ホーム

    # # ピッチャーマウンドを描く（中央円）
    # cv2.circle(field, center, 50, (0, 0, 255), 2)

    # # 外野を描画（ダイヤモンド形状）
    # cv2.line(field, (center[0] - 200, center[1]), (center[0], center[1] + 200), (0, 255, 0), 2)
    # cv2.line(field, (center[0] + 200, center[1]), (center[0], center[1] + 200), (0, 255, 0), 2)

    # # 外野フェンス（長方形として描く）
    # cv2.rectangle(field, (center[0] - 250, center[1] - 250), (center[0] + 250, center[1] + 250), (0, 0, 255), 3)

    # # ホームプレートを描く（五角形）
    # home_plate_points = np.array([(center[0] - 10, center[1] + 200), (center[0] + 10, center[1] + 200),
    #                             (center[0] + 10, center[1] + 180), (center[0], center[1] + 160),
    #                             (center[0] - 10, center[1] + 180)], np.int32)
    # cv2.polylines(field, [home_plate_points], isClosed=True, color=(0, 0, 255), thickness=2)

    # # ピッチャーマウンドとホームベースを結ぶ線（ピッチャーとホーム間の線）
    # cv2.line(field, center, (center[0], center[1] + 200), (0, 0, 0), 2)

    # # バッターボックスを描く（長方形）
    # cv2.rectangle(field, (center[0] - 40, center[1] + 160), (center[0] + 40, center[1] + 180), (255, 0, 0), 2)

    # # 画像を表示
    # cv2.imshow('Baseball Field', field)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    print(len(df_master))
