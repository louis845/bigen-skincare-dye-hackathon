import dotenv
import uvicorn

import skincare.main

dotenv.load_dotenv()

if __name__ == "__main__":
    uvicorn.run(skincare.main.demo.app, host="0.0.0.0", port=80)