export async function sseConnection(token: string) {
    const res = await fetch("http://localhost:3000/user-management/sse", {
        method: 'GET',
        headers: {
            'Content-Type': 'text/event-stream',
            'Authorization': `Bearer ${token}`
        }
    })

    const reader = res.body?.pipeThrough(new TextDecoderStream()).getReader() ?? null;
    while (reader) {
        const {value, done} = await reader.read();
        if (done) break;
        if (value.startsWith('retry: ')) continue;
        const parse = JSON.parse(value?.replace('data: ', ''));
        console.log(parse);
        sseHandler(parse.event, parse.data);
    }
}

export function sseHandler(process: string, data: any ) {
    if (process == "invite") {
        console.log(data,  " et ", process);
    }
}
