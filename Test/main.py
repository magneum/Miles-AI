import pygame
import subprocess
import pyttsx3


def play_audio_from_url(audio_url, volume=0.5):
    try:
        # Set the output file name
        output_file = "output.mp3"

        # Download the audio using ffmpeg
        subprocess.call(["bin/ffmpeg", "-i", audio_url, output_file])

        # Initialize Pygame
        pygame.init()
        pygame.display.set_caption("Audio Player")

        # Load the audio file into a Pygame mixer
        pygame.mixer.init()
        pygame.mixer.music.load(output_file)

        # Set the initial volume
        pygame.mixer.music.set_volume(volume)

        # Play the audio file
        pygame.mixer.music.play()

        # Initialize pyttsx3
        engine = pyttsx3.init()

        # Loop to handle events and keep the audio playing
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        # Pause or unpause the audio
                        if pygame.mixer.music.get_busy():
                            pygame.mixer.music.pause()
                            engine.say("Paused")
                        else:
                            pygame.mixer.music.unpause()
                            engine.say("Resumed")
                        engine.runAndWait()
                    elif event.key == pygame.K_s:
                        # Stop the audio and exit the loop
                        pygame.mixer.music.stop()
                        engine.say("Stopped")
                        engine.runAndWait()
                        return
                    elif event.key == pygame.K_UP:
                        # Increase the volume by 10%
                        volume = min(
                            pygame.mixer.music.get_volume() + 0.1, 1.0)
                        pygame.mixer.music.set_volume(volume)
                        engine.say("Volume increased")
                        engine.runAndWait()
                    elif event.key == pygame.K_DOWN:
                        # Decrease the volume by 10%
                        volume = max(
                            pygame.mixer.music.get_volume() - 0.1, 0.0)
                        pygame.mixer.music.set_volume(volume)
                        engine.say("Volume decreased")
                        engine.runAndWait()

    except subprocess.CalledProcessError as e:
        print("Error: Could not download audio from URL:", audio_url)
        print("Error message:", e.output)
        return

    except pygame.error as e:
        print("Error: Pygame mixer could not load audio file:", output_file)
        print("Error message:", e)
        return

    except KeyboardInterrupt:
        # Handle the user pressing Ctrl-C to exit
        print("Exiting due to keyboard interrupt")
        pygame.quit()
        quit()

    except Exception as e:
        # Handle any other unexpected errors
        print("Unexpected error occurred:", e)
        return


audio_url = "https://rr2---sn-gwpa-jj0z.googlevideo.com/videoplayback?expire=1679857267&ei=E0IgZMmdK7WL1d8P1rWc4Ak&ip=136.232.89.206&id=o-AIXYfvlw0fq6ElK15J7Butn3Zf6FhFpS_T3coLsS_YnP&itag=140&source=youtube&requiressl=yes&mh=43&mm=31%2C29&mn=sn-gwpa-jj0z%2Csn-gwpa-h55e7&ms=au%2Crdu&mv=m&mvi=2&pcm2cms=yes&pl=20&initcwndbps=433750&spc=99c5CTReMtmva9H2DVkvQ5KVSVvCjbo&vprv=1&svpuc=1&mime=audio%2Fmp4&gir=yes&clen=45337782&dur=2801.371&lmt=1673939934438767&mt=1679835185&fvip=7&keepalive=yes&fexp=24007246&c=ANDROID&txp=4532434&sparams=expire%2Cei%2Cip%2Cid%2Citag%2Csource%2Crequiressl%2Cspc%2Cvprv%2Csvpuc%2Cmime%2Cgir%2Cclen%2Cdur%2Clmt&sig=AOq0QJ8wRgIhAMVbuRAkIBCWBda_X2lfLvMCC_hsCbnhnkDPWUgFdWuUAiEAwdXt2gDswiQ8_5nxzHSSWbSKMZjG92C5r8DsMRnpwaY%3D&lsparams=mh%2Cmm%2Cmn%2Cms%2Cmv%2Cmvi%2Cpcm2cms%2Cpl%2Cinitcwndbps&lsig=AG3C_xAwRQIgUhMS3mFalovWhVjMweXXjWe-pLPs6pL1anIjkALNlKACIQDZVMFkTVvppNhoyPmjqqbkuxlp352wgy-aUlbuTsqhAw%3D%3D"
volume = input("Enter volume value (0.0-1.0): ")
try:
    volume = float(volume)
    if volume < 0 or volume > 1:
        raise ValueError
except ValueError:
    print("Invalid volume value, using default value of 0.5")
    volume = 0.5

play_audio_from_url(audio_url, volume=volume)
