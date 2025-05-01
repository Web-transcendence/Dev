

export async function sseConnection() {
    try {
        const token = localStorage.getItem('token')
        const res = await fetch(`/user-management/sse`, {
            method: 'GET',
            headers: {
                'Content-Type': 'text/event-stream',
                'Authorization': `Bearer ${token}`
            }
        })
        if (res.status === 100)
            return;
        if (!res.ok) {
            const error = res.json()
            console.log(error)
            //notifyhandling
            return ;
        }
        console.log('sse connection')
            const reader = res.body?.pipeThrough(new TextDecoderStream()).getReader() ?? null;
        while (reader) {
            const {value, done} = await reader.read();
            if (done) break;
            if (value.startsWith('retry: ')) continue;
            const parse = JSON.parse(value?.replace('data: ', ''));
            sseHandler(parse.event, parse.data);
        }
    } catch (err) {
        console.error(err);
    }
}

export function sseHandler(process: string, data: any ) {
    if (process == "invite") {
        console.log(data,  " et ", process);
    }
}


