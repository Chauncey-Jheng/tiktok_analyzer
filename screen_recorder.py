import argparse 
from subprocess import call
from sys import argv

def main():
   parser = argparse.ArgumentParser("Recording video and audio using ffmpeg\n")
   parser.add_argument("output", help="Set output file")
   parser.add_argument("--vcodec", help="Set video codec")
   parser.add_argument("--acodec", help="Set the audio codec. Setting this will enable audio, if it is not set no audio will be recorded")
   parser.add_argument("-ox", "--offsetx", type=int, default=0, help="Set the x offset")
   parser.add_argument("-oy", "--offsety", type=int, default=0, help="Set the y offset")
   parser.add_argument("-r", "--fps", type=int, default=30, help="Set the fps")
   parser.add_argument("-W", "--width", type=int, default=1900, help="Set the width of the recording area")
   parser.add_argument("-H", "--height", type=int, default=1000, help="Set the height of the recording area")
   parser.add_argument("-ac", "--audio_channels", type=int, default=2, help="Set the audio channels")
   parser.add_argument("-t","--duration_time", type=int, default=20, help="Set the duration time")
   args = parser.parse_args()
   record(args)
   
def record(args):
   output="static/video/record_video" + args.output
   vcodec=args.vcodec
   acodec=args.acodec
   offsetx=str(args.offsetx)
   offsety=str(args.offsety)
   fps=str(args.fps)
   width=str(args.width)
   height=str(args.height)
   ac=str(args.audio_channels)
   time = str(args.duration_time)
   print(args)
   
   cmdstr = "ffmpeg -y --enable-libxcb -f xcbgrab -s " + width + "x" + height + "  -i :0.0+" + offsetx + "," + offsety + " -ac " + ac + " -f alsa -i pulse -acodec libmp3lame " + " -t " + time + " " + output
   call(cmdstr, shell=True)


if __name__ == "__main__":
    main()