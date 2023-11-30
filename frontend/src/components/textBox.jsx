import { Textarea } from '@chakra-ui/react'
const TextBox = ({text, setText}) => {
    return (<>
        <Textarea 
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder='Enter the news article here'  height='200px' background={'blue.50'} resize='vertical' />
    </>)
}

export default TextBox;