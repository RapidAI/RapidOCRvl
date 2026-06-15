export namespace main {
	
	export class OCRRequest {
	    apiUrl: string;
	    apiKey: string;
	    imagePath: string;
	    task: string;
	    maxNewTokens: number;
	    timeoutSecs: number;
	
	    static createFrom(source: any = {}) {
	        return new OCRRequest(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.apiUrl = source["apiUrl"];
	        this.apiKey = source["apiKey"];
	        this.imagePath = source["imagePath"];
	        this.task = source["task"];
	        this.maxNewTokens = source["maxNewTokens"];
	        this.timeoutSecs = source["timeoutSecs"];
	    }
	}
	export class OCRResult {
	    text: string;
	    promptTokens: number;
	    generatedTokens: number;
	    raw: string;
	
	    static createFrom(source: any = {}) {
	        return new OCRResult(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.text = source["text"];
	        this.promptTokens = source["promptTokens"];
	        this.generatedTokens = source["generatedTokens"];
	        this.raw = source["raw"];
	    }
	}
	export class ReadyResult {
	    status: string;
	    backend: string;
	    quantization: string;
	    weightSource: string;
	    visionLoaded: boolean;
	    concurrency: number;
	    availableSlots: number;
	    raw: string;
	
	    static createFrom(source: any = {}) {
	        return new ReadyResult(source);
	    }
	
	    constructor(source: any = {}) {
	        if ('string' === typeof source) source = JSON.parse(source);
	        this.status = source["status"];
	        this.backend = source["backend"];
	        this.quantization = source["quantization"];
	        this.weightSource = source["weightSource"];
	        this.visionLoaded = source["visionLoaded"];
	        this.concurrency = source["concurrency"];
	        this.availableSlots = source["availableSlots"];
	        this.raw = source["raw"];
	    }
	}

}

