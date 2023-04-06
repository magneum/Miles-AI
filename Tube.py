# import os
# import spacy
# import random
# from spacy.training import Example
# from spacy.util import minibatch, compounding


# trainData = [
#     (
#         "I want to listen to 'Bohemian Rhapsody' by Queen",
#         {"entities": [(20, 37, "SONG")]},
#     ),
#     (
#         "Can you play 'Stairway to Heaven' by Led Zeppelin?",
#         {"entities": [(16, 35, "SONG")]},
#     ),
#     (
#         "I'm in the mood for some 'Billie Jean' by Michael Jackson",
#         {"entities": [(27, 39, "SONG")]},
#     ),
#     (
#         "Can you play the song Bad Guy by Billie Eilish?",
#         {"entities": [(22, 30, "SONG"), (35, 47, "ARTIST")]},
#     ),
#     (
#         "Can you play some music by Taylor Swift?",
#         {"entities": [(21, 26, "SONG"), (30, 42, "ARTIST")]},
#     ),
#     (
#         "I want to listen to Imagine Dragons' Thunder",
#         {"entities": [(27, 34, "SONG"), (0, 16, "ARTIST")]},
#     ),
#     ("Find me a video of the song Despacito", {"entities": [(26, 34, "SONG")]}),
#     (
#         "Can you play the song Hotel California by Eagles?",
#         {"entities": [(22, 38, "SONG"), (42, 48, "ARTIST")]},
#     ),
#     (
#         "What is the name of the song playing right now?",
#         {"entities": [(28, 32, "SONG")]},
#     ),
#     (
#         "I really like the song Uptown Funk by Mark Ronson",
#         {"entities": [(22, 32, "SONG"), (36, 47, "ARTIST")]},
#     ),
#     ("Play some songs by Adele", {"entities": [(17, 22, "SONG"), (26, 31, "ARTIST")]}),
#     (
#         "Can you find the song Stairway to Heaven by Led Zeppelin?",
#         {"entities": [(21, 39, "SONG"), (43, 56, "ARTIST")]},
#     ),
#     (
#         "I want to listen to the song Shake It Off by Taylor Swift",
#         {"entities": [(27, 38, "SONG"), (42, 54, "ARTIST")]},
#     ),
#     (
#         "Can you play me a video of the song Bohemian Rhapsody?",
#         {"entities": [(36, 55, "SONG")]},
#     ),
#     (
#         "Find me some music by Post Malone",
#         {"entities": [(16, 22, "SONG"), (26, 37, "ARTIST")]},
#     ),
#     (
#         "I can't get this song out of my head, it's called Can't Stop the Feeling by Justin Timberlake",
#         {"entities": [(49, 67, "SONG"), (71, 88, "ARTIST")]},
#     ),
#     (
#         "Can you play the song Enter Sandman by Metallica?",
#         {"entities": [(22, 35, "SONG"), (39, 48, "ARTIST")]},
#     ),
#     (
#         "What is the song that goes 'do you ever feel like a plastic bag'?",
#         {"entities": [(20, 45, "SONG")]},
#     ),
#     (
#         "I want to listen to some music by Beyonce",
#         {"entities": [(24, 29, "SONG"), (33, 39, "ARTIST")]},
#     ),
#     (
#         "Can you play the song Purple Rain by Prince?",
#         {"entities": [(22, 33, "SONG"), (37, 43, "ARTIST")]},
#     ),
#     (
#         "Find me the video for the song Sweet Child O' Mine",
#         {"entities": [(33, 54, "SONG")]},
#     ),
#     (
#         "I really love the song Rolling in the Deep by Adele",
#         {"entities": [(20, 37, "SONG"), (41, 46, "ARTIST")]},
#     ),
#     (
#         "Can you play some songs by Maroon 5?",
#         {"entities": [(17, 22, "SONG"), (26, 32, "ARTIST")]},
#     ),
#     (
#         "Play the song Stairway to Heaven by Led Zeppelin",
#         {"entities": [(17, 36, "SONG"), (40, 54, "ARTIST")]},
#     ),
#     (
#         "Can you play Hotel California by Eagles?",
#         {"entities": [(17, 33, "SONG"), (37, 43, "ARTIST")]},
#     ),
#     (
#         "I want to listen to Bohemian Rhapsody by Queen",
#         {"entities": [(21, 39, "SONG"), (43, 48, "ARTIST")]},
#     ),
#     (
#         "Play the song Sweet Child O' Mine by Guns N' Roses",
#         {"entities": [(17, 40, "SONG"), (44, 57, "ARTIST")]},
#     ),
#     (
#         "Can you play the song Thriller by Michael Jackson?",
#         {"entities": [(21, 28, "SONG"), (32, 48, "ARTIST")]},
#     ),
#     (
#         "I want to listen to the song Billie Jean by Michael Jackson",
#         {"entities": [(23, 33, "SONG"), (37, 53, "ARTIST")]},
#     ),
#     (
#         "Play the song Smells Like Teen Spirit by Nirvana",
#         {"entities": [(17, 44, "SONG"), (48, 54, "ARTIST")]},
#     ),
#     (
#         "Can you play Thunderstruck by AC/DC?",
#         {"entities": [(15, 27, "SONG"), (31, 36, "ARTIST")]},
#     ),
#     (
#         "I want to listen to the song Don't Stop Believin' by Journey",
#         {"entities": [(23, 43, "SONG"), (47, 53, "ARTIST")]},
#     ),
#     (
#         "Play the song November Rain by Guns N' Roses",
#         {"entities": [(17, 31, "SONG"), (35, 48, "ARTIST")]},
#     ),
#     (
#         "Can you play Purple Haze by Jimi Hendrix?",
#         {"entities": [(15, 26, "SONG"), (30, 42, "ARTIST")]},
#     ),
#     (
#         "I want to listen to the song Imagine by John Lennon",
#         {"entities": [(23, 30, "SONG"), (34, 45, "ARTIST")]},
#     ),
#     (
#         "Play the song Enter Sandman by Metallica",
#         {"entities": [(17, 32, "SONG"), (36, 45, "ARTIST")]},
#     ),
#     (
#         "Can you play Nothing Else Matters by Metallica?",
#         {"entities": [(15, 35, "SONG"), (39, 48, "ARTIST")]},
#     ),
#     (
#         "I want to listen to the song Like a Rolling Stone by Bob Dylan",
#         {"entities": [(23, 44, "SONG"), (48, 57, "ARTIST")]},
#     ),
#     (
#         "Play the song Walk This Way by Aerosmith",
#         {"entities": [(17, 30, "SONG"), (34, 43, "ARTIST")]},
#     ),
#     (
#         "Can you play Livin' on a Prayer by Bon Jovi?",
#         {"entities": [(15, 32, "SONG"), (36, 44, "ARTIST")]},
#     ),
#     (
#         "I want to listen to the song The Sound of Silence by Simon & Garfunkel",
#         {"entities": [(23, 43, "SONG"), (47, 63, "ARTIST")]},
#     ),
#     (
#         "Play the song Shape of You by Ed Sheeran",
#         {"entities": [(13, 26, "SONG"), (31, 42, "ARTIST")]},
#     ),
#     ("Play some music by Queen", {"entities": [(16, 21, "GENRE"), (25, 30, "ARTIST")]}),
#     (
#         "Can you play the song Viva la Vida by Coldplay?",
#         {"entities": [(19, 32, "SONG"), (37, 45, "ARTIST")]},
#     ),
#     (
#         "I want to hear the song Bohemian Rhapsody by Queen",
#         {"entities": [(19, 35, "SONG"), (40, 45, "ARTIST")]},
#     ),
#     (
#         "Play some hip-hop music by Kendrick Lamar",
#         {"entities": [(16, 23, "GENRE"), (27, 41, "ARTIST")]},
#     ),
#     (
#         "Can you play the song Bad Guy by Billie Eilish?",
#         {"entities": [(19, 26, "SONG"), (31, 43, "ARTIST")]},
#     ),
#     ("I'm in the mood for some classical music", {"entities": [(29, 38, "GENRE")]}),
#     (
#         "Can you play the song Sweet Child O' Mine by Guns N' Roses?",
#         {"entities": [(19, 40, "SONG"), (45, 58, "ARTIST")]},
#     ),
#     (
#         "I want to hear some jazz music by Louis Armstrong",
#         {"entities": [(19, 23, "GENRE"), (27, 41, "ARTIST")]},
#     ),
#     (
#         "Play the song Stairway to Heaven by Led Zeppelin",
#         {"entities": [(13, 32, "SONG"), (37, 49, "ARTIST")]},
#     ),
#     (
#         "Can you play some country music by Johnny Cash?",
#         {"entities": [(19, 26, "GENRE"), (30, 41, "ARTIST")]},
#     ),
#     (
#         "I'm in the mood for the song Yesterday by The Beatles",
#         {"entities": [(30, 38, "SONG"), (43, 55, "ARTIST")]},
#     ),
#     (
#         "Play some rock music by AC/DC",
#         {"entities": [(16, 20, "GENRE"), (24, 29, "ARTIST")]},
#     ),
#     (
#         "Can you play the song Purple Rain by Prince?",
#         {"entities": [(19, 31, "SONG"), (36, 42, "ARTIST")]},
#     ),
#     (
#         "I want to hear some pop music by Ariana Grande",
#         {"entities": [(19, 22, "GENRE"), (26, 39, "ARTIST")]},
#     ),
#     (
#         "Play the song Hotel California by The Eagles",
#         {"entities": [(13, 31, "SONG"), (36, 47, "ARTIST")]},
#     ),
#     (
#         "Can you play some electronic music by Daft Punk?",
#         {"entities": [(19, 29, "GENRE"), (33, 42, "ARTIST")]},
#     ),
#     (
#         "I'm in the mood for the song Don't Stop Believin' by Journey",
#         {"entities": [(30, 49, "SONG"), (54, 61, "ARTIST")]},
#     ),
#     (
#         "Play some indie music by Arctic Monkeys",
#         {"entities": [(16, 20, "GENRE"), (24, 37, "ARTIST")]},
#     ),
#     (
#         "Can you play Watermelon Sugar by Harry Styles?",
#         {"entities": [(13, 29, "SONG"), (33, 45, "ARTIST")]},
#     ),
#     (
#         "I'm in the mood for some Bohemian Rhapsody",
#         {
#             "entities": [
#                 (30, 49, "SONG"),
#             ]
#         },
#     ),
#     (
#         "Please put on Smooth by Santana",
#         {"entities": [(16, 22, "SONG"), (26, 33, "ARTIST")]},
#     ),
#     (
#         "I heard that Shape of You by Ed Sheeran is popular",
#         {"entities": [(20, 30, "SONG"), (34, 44, "ARTIST")]},
#     ),
#     (
#         "I want to listen to Billie Jean by Michael Jackson",
#         {"entities": [(23, 34, "SONG"), (38, 54, "ARTIST")]},
#     ),
#     (
#         "Have you heard of Hey Jude by The Beatles?",
#         {"entities": [(17, 25, "SONG"), (29, 40, "ARTIST")]},
#     ),
#     (
#         "I'm looking for Somebody That I Used to Know by Gotye",
#         {"entities": [(25, 50, "SONG"), (54, 59, "ARTIST")]},
#     ),
#     (
#         "Can you play Thriller by Michael Jackson?",
#         {"entities": [(13, 21, "SONG"), (25, 41, "ARTIST")]},
#     ),
#     (
#         "I'm in the mood for some Let It Be by The Beatles",
#         {"entities": [(27, 35, "SONG"), (39, 50, "ARTIST")]},
#     ),
#     (
#         "Please put on Hotel California by Eagles",
#         {"entities": [(16, 33, "SONG"), (37, 43, "ARTIST")]},
#     ),
#     (
#         "I heard that Stairway to Heaven by Led Zeppelin is a classic",
#         {"entities": [(20, 38, "SONG"), (42, 55, "ARTIST")]},
#     ),
#     (
#         "I want to listen to Lose Yourself by Eminem",
#         {"entities": [(23, 36, "SONG"), (40, 46, "ARTIST")]},
#     ),
#     (
#         "Have you heard of Livin' on a Prayer by Bon Jovi?",
#         {"entities": [(17, 37, "SONG"), (41, 49, "ARTIST")]},
#     ),
#     (
#         "I'm looking for some Purple Rain by Prince",
#         {"entities": [(22, 33, "SONG"), (37, 43, "ARTIST")]},
#     ),
#     (
#         "Can you play November Rain by Guns N' Roses?",
#         {"entities": [(13, 27, "SONG"), (31, 45, "ARTIST")]},
#     ),
#     (
#         "I'm in the mood for some Nothing Else Matters by Metallica",
#         {"entities": [(30, 51, "SONG"), (55, 64, "ARTIST")]},
#     ),
#     (
#         "Please put on Fly Me to the Moon by Frank Sinatra",
#         {"entities": [(16, 36, "SONG"), (40, 53, "ARTIST")]},
#     ),
#     (
#         "I heard that Smells Like Teen Spirit by Nirvana is a grunge classic",
#         {"entities": [(20, 41, "SONG"), (45, 51, "ARTIST")]},
#     ),
# ]


# nlp = spacy.load("en_core_web_md")
# ner = nlp.get_pipe("ner")
# ner.add_label("SONG")

# epochs = 100
# dropout = 0.5
# batch_size = 8
# best_accuracy = 0.0
# best_loss = float("inf")
# optimizer = nlp.begin_training()

# disable_pipes = [pipe for pipe in nlp.pipe_names if pipe != "ner"]
# with nlp.disable_pipes(*disable_pipes):
#     for epoch in range(epochs):
#         random.shuffle(trainData)
#         losses = {}
#         batches = minibatch(trainData, size=compounding(4.0, 32.0, 1.001))
#         for batch in batches:
#             example_batch = []
#             for text, annotations in batch:
#                 example = Example.from_dict(nlp.make_doc(text), annotations)
#                 example_batch.append(example)
#             nlp.update(example_batch, drop=dropout, sgd=optimizer, losses=losses)
#         print(f"Epoch {epoch+1} Loss: {losses['ner']:.3f}")


# nlp.to_disk("song_ner_model")

# while True:
#     user_input = input("Enter a sentence or 'quit' to exit: ")
#     if user_input.lower() == "quit":
#         break
#     doc = nlp(user_input)
#     for ent in doc.ents:
#         print(f"{ent.label_.upper()}: {ent.text}")
