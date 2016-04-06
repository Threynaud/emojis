from emojis_lib import *
import time
from requests.exceptions import Timeout, ConnectionError
from requests.packages.urllib3.exceptions import ReadTimeoutError, ProtocolError
from http.client import IncompleteRead
import ssl

while True:
    try:
        stream_emojis(track=["the", "and", "I", "you", "he", "we", "do", "does", "they", "have", "has", "am", "are", "me", "him", "her",
           "it", "us", "them", "can", "can't","is","but", "a", "so", "that", "this", "yes", "no", "very", "I'm",
           "you're", "he's", "she's", "we're", "they're", "there", "did", "had", "hadn't", "don't", "out", "in",
           "about", "being", "for", "having", "who", "when", "what", "where", "of", "on", "🐶", "🐱", "🐭", "🐹", "🐰", "🐻", "🐼", "🐨", "🐯", "🦁", "🐮", "🐷", "🐽", "🐸", "🐙", "🐵", "🙈",
          "🙉", "🙊", "🐒", "🐔", "🐧", "🐦", "🐤", "🐣", "🐥", "🐺", "🐗", "🐴", "🦄", "🐝", "🐛", "🐌", "🐞",
          "🐜", "🕷", "🦂", "🦀", "🐍", "🐢", "🐠", "🐡", "🐬", "🐳", "🐋", "🐊", "🐆", "🐅", "🐃", "🐂", "🐄",
          "🐪", "🐫", "🐘", "🐐", "🐏", "🐑", "🐎", "🐖", "🐀", "🐁", "🐓", "🦃", "🕊", "🐕", "🐩", "🐈", "🐇",
          "🐿","🐾", "🐉", "🐲", "🌵", "🎄", "🌲", "🌳", "🌴", "🌱", "🌿", "☘", "🍀", "🎍", "🎋", "🍃", "🍂",
          "🍁", "🌾", "🌺", "🌻", "🌹", "🌷", "🌼", "🌸", "💐", "🍄", "🌰", "🎃", "🐚", "🕸", "🌎", "🌍", "🌏",
          "🌕", "🌖", "🌗", "🌘", "🌑", "🌒", "🌓", "🌔", "🌚", "🌝", "🌛", "🌜", "🌞", "🌙", "⭐️", "🌟", "💫",
          "✨", "☄️", "☀️", "🌤", "⛅️", "🌥", "🌦", "☁️", "🌧", "⛈", "🌩", "⚡️", "🔥", "💥", "❄️", "🌨", "☃️", "⛄️",
          "🌬", "💨", "🌪", "🌫", "☂️", "☔️", "💧", "💦", "🌊", "🍓","🍈", "🍒", "🍑", "🍍", "🍅", "🍆", "🌶", "🌽", "🍠", "🍯", "🍞", "🧀", "🍗", "🍖", "🍤", "🍳", "🍔",
       "🍟", "🌭", "🍕", "🍝", "🌮", "🌯", "🍜", "🍲", "🍥", "🍣", "🍱", "🍛", "🍙", "🍚", "🍘", "🍢", "🍡", "🍧",
       "🍨", "🍦", "🍰", "🎂", "🍮", "🍬", "🍭", "🍫", "🍿", "🍩", "🍪", "🍺", "🍻", "🍷", "🍸", "🍹", "🍾", "🍶",
       "🍵", "☕️", "🍼", "🍴", "🍽","⚽️", "🏀", "🏈", "⚾️", "🎾", "🏐", "🏉", "🎱", "⛳️", "🏌", "🏓", "🏸", "🏒", "🏑",
          "🏏", "🎿", "⛷", "🏂", "⛸", "🏹", "🎣", "🚣", "🏊", "🏄", "🛀", "⛹", "🏋", "🚴", "🚵", "🏇", "🕴", "🏆",
          "🎽", "🏅", "🎖", "🎗", "🏵", "🎫", "🎟", "🎭", "🎨", "🎪", "🎤", "🎧", "🎼", "🎹", "🎷", "🎺", "🎸", "🎻",
          "🎬", "🎮", "👾", "🎯", "🎲", "🎰", "🎳"],
                      listener=create_listener("english3"), auth=authentify(".twitter_config"))
    except (Timeout, ssl.SSLError, ReadTimeoutError, ConnectionError, ProtocolError, IncompleteRead) as exc:
        print("Disconnected, reconnection in 7 seconds")
        time.sleep(7)
