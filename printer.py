class Printer:
    
    activated = False
    
    def print(*args):
        if not Printer.activated:
            return
        print(*args)