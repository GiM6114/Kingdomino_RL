class Printer:
    
    activated = False
    
    def print(self, *args):
        if not Printer.activated:
            return
        print(*args)