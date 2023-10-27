import RPi.GPIO as GPIO
import time
GPIO.setmode(GPIO.BCM)
RR_PIN=17
RG_PIN=27
RB_PIN=22
LR_PIN=5
LG_PIN=6
LB_PIN=26
GPIO.setup([RR_PIN,RG_PIN,RB_PIN,LR_PIN,LG_PIN,LB_PIN], GPIO.OUT)

def main():
    try:
        while True:
            GPIO.output(RR_PIN, GPIO.HIGH)
            time.sleep(1)
            GPIO.output(RR_PIN, GPIO.LOW)
            GPIO.output(RG_PIN, GPIO.HIGH)
            time.sleep(1)
            GPIO.output(RG_PIN, GPIO.LOW)
            GPIO.output(RB_PIN, GPIO.HIGH)
            time.sleep(1)
            GPIO.output(RB_PIN, GPIO.LOW)

            GPIO.output(LR_PIN, GPIO.HIGH)
            time.sleep(1)
            GPIO.output(LR_PIN, GPIO.LOW)
            GPIO.output(LG_PIN, GPIO.HIGH)
            time.sleep(1)
            GPIO.output(LG_PIN, GPIO.LOW)
            GPIO.output(LB_PIN, GPIO.HIGH)
            time.sleep(1)
            GPIO.output(LB_PIN, GPIO.LOW)
            # GPIO.output([R_PIN, G_PIN], GPIO.HIGH) #yellow
            # time.sleep(1)
            # GPIO.output([R_PIN, G_PIN], GPIO.LOW)

            # GPIO.output([G_PIN, B_PIN], GPIO.HIGH) #water
            # time.sleep(1)
            # GPIO.output([G_PIN, B_PIN], GPIO.LOW)

            # GPIO.output([R_PIN, B_PIN], GPIO.HIGH) #purple
            # time.sleep(1)
            # GPIO.output([R_PIN, B_PIN], GPIO.LOW)

            # GPIO.output([R_PIN, B_PIN, G_PIN], GPIO.HIGH) #white
            # time.sleep(1)
            # GPIO.output([R_PIN, B_PIN, G_PIN], GPIO.LOW)

    except KeyboardInterrupt:
            GPIO.output([RR_PIN, RG_PIN, RB_PIN, LR_PIN, LG_PIN, LB_PIN], GPIO.LOW)
            GPIO.cleanup()


if __name__=="__main__":
    main()