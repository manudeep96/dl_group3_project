import { Textarea } from '@chakra-ui/react'
const TextBox = ({text, setText}) => {
    return (<>
        <Textarea 
        value={text}
        onChange={(e) => setText(e.target.value)}
        placeholder='Enter the news article here' margin={8} height='200px' w={'70%'} background={'white'} resize='vertical' />
    </>)
}

export default TextBox;