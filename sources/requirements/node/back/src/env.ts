import {env as nodeEnv} from 'node:process';
import {z} from 'zod';

const zEnv = z.object({
    TRANS_FRONT_PATH: z.string().default('../../front/dist/'),
    TRANS_VIEWS_PATH: z.string().default('../src/views/'),
    LAST_URL: z.string().default(''),
})
const env = zEnv.parse(nodeEnv);

export {env}