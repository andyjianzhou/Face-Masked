machineLearning, webDev, uiUx, blockchain, other = 0, 0, 0, 0, 0
exit = True
def poll(machineLearning, webDev, uiUx, blockchain, other, exit):
    print("1. Machine Learning\n2. Web Development\n3. UI/UX\n4. Blockchain\n5. Other")
    while(exit):
        userAnswer = int(input("What do you want to learn: "))
        if userAnswer == 1:
            machineLearning += 1
        elif userAnswer == 2:
            webDev += 1
        elif userAnswer == 3:
            uiUx += 1
        elif userAnswer == 4:
            blockchain += 1
        elif userAnswer == 5:
            other += 1
        elif userAnswer == 0:
            exit = False
    return machineLearning, webDev, uiUx, blockchain, other
machineLearning, webDev, uiUx, blockchain, other = poll(machineLearning, webDev, uiUx, blockchain, other, exit)
print(f"Machine Learning: {machineLearning}\nWeb Development: {webDev}\nUI/UX: {uiUx}\nBlockchain: {blockchain}\nOther: {other}")

        
